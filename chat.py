import random, logging
from itertools import combinations
from autogen import ConversableAgent, GroupChat, GroupChatManager, config_list_from_json

from utils import * 


logging.basicConfig(level=logging.INFO, format='%(message)s')

class Chat:

    def __init__(self, args):

        self.threshold = args.threshold
        self.group_size = args.group_size
        self.llm_config = {"config_list": config_list_from_json(env_or_file=args.config_file_or_env)}
        self.assistant = ConversableAgent("assistant", llm_config=self.llm_config)
        self.agents = []
        
        self.onboarding = []
        self.discussion = []
        self.reflection = []
        self.opinion = []
        self.region = []

        self.exception = None
        self.prediction = None
        self.status = None
        self.results = {}
    
    @property
    def gold_agents(self): 
       return [agent for agent in  self.agents if agent.is_gold]

    @property
    def selected_agents(self): 
       return [agent for agent in  self.agents if agent.is_selected]
            
    """agent creation and onboarding"""

    def create_agents(self, names, labels, system_messages): 
        for name,  system_message in zip(names, system_messages) :
            self.agents.append(
                Agent(name, system_message, labels[name], self.llm_config)
            )

    def interview_agent(self, agent, question, chars):
        chat = self.assistant.initiate_chat(
            agent,
            message=question,
            max_turns=1,
            coding=False
        )
        reply = chat.chat_history[1]['content']
        opinion = pred_extract(reply, chars)
        return opinion
    
    def onboard_agents(self, chars, onboarding_question):   
        for agent in self.agents:
            opinion = self.interview_agent(agent.agent, onboarding_question, chars)
            agent.save_onboarding(opinion, self.threshold)

    def create_onboarded_agents(self): 
        for agent in self.selected_agents:
            agent.add_onboarding()

    """agent reflection"""

    def process_discussion(self, discussion): 
        return 'Given the following discussion:\n\n'+'\n\n'.join([message['name']+': '+message['content'] for message in discussion])

    def reflect_agents(self, chars, reflection_question):
        discussion = self.process_discussion(self.discussion)
        for agent in self.selected_agents:
            opinion = self.interview_agent(agent.agent, discussion+'\n\n'+reflection_question, chars)
            agent.reflection = opinion

    """agent selection"""

    def select_agents(self, counts, filter):
        # Acquiring the filtered features based on the provided filter name
        features = [getattr(agent, filter) for agent in self.gold_agents]
        # Generating and evaluating combinations
        combs, entropy = self.evaluate_combinations(features)
        # Selecting a combination with the entropy of the lowest count
        comb, choice = self.select_combination(combs, entropy, counts)
        # Selecting agents based on the chosen combination
        self.assign_selected_agents(self.gold_agents, comb, features)
        return choice

    def evaluate_combinations(self, features):
        combs = list(set([tuple(sorted(c)) for c in combinations(features, self.group_size)]))
        entropy = np.array([round(get_entropy(c), 2) for c in combs])
        return combs, entropy

    def select_combination(self, combs, entropy, counts):
        valid_entropy_counts =  {k: v for k, v in counts.items() if k in set(entropy)}
        least_frequent_entropy = min(valid_entropy_counts, key=valid_entropy_counts.get)
        selected_index = random.choice(np.where(entropy == least_frequent_entropy)[0])
        counts[least_frequent_entropy] += 1
        return combs[selected_index], least_frequent_entropy

    def assign_selected_agents(self, agents, comb, features):
        for feature in comb:
            ind = random.choice(np.where(np.array(features) == feature)[0])
            agents[ind].is_selected = True
            features[ind] = None

    """agent group discussion"""

    def group_chat(self, chars, discussion_question, group_opinion_summary_args, agent_reply_summary_prompt): 
        agents = [agent.agent for agent in self.selected_agents]
        groupchat = GroupChat(agents=agents, messages=[], max_round=15)
        manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)
        assistant = ConversableAgent("groupchat_assistant", llm_config=self.llm_config)
        chat = assistant.initiate_chat(
            recipient=manager,
            message=discussion_question,
            summary_method="reflection_with_llm",
            summary_args=group_opinion_summary_args,
            max_turns=1,
            code_execution_config=False
            )
        opinion = pred_extract(chat.summary, chars)
        self.prediction = {'summary': chat.summary, 'opinion': opinion}
        for message in groupchat.messages[1:]: 
            summary = assistant.generate_reply(messages=[{"content": f"{message['content']}\n\n{agent_reply_summary_prompt}", "role": "user"}])
            opinion = pred_extract(summary, chars)
            message.update({'summary': summary, "opinion": opinion})
            logging.info(f"Summary     : {summary}\nOpinion:     : {opinion}")
            self.discussion.append(message)
            
    """saving results"""

    def add_exception(self, e): 
        self.exception=str(e)
    
    def results_json(self, data, entropy):
        # TODO: use agents
        return {
            'onboarding_entropy': entropy, 
            'example': data, 
            'onboarding': {agent.agent.name: agent.opinion for agent in self.selected_agents}, 
            'discussion': self.discussion,
            'group_opinion': self.prediction,
            'reflection':{agent.agent.name: agent.reflection for agent in self.selected_agents}, 
            'exception': self.exception,
        }


class Agent:

    def __init__(self, name, system_message, label, llm_config): 
        self.is_gold = False
        self.is_selected = False
        self.opinion = None
        self.reflection = None
        self.label = label
        self.agent = ConversableAgent(
                name=name,
                system_message=system_message,
                llm_config=llm_config
        )
        
    @property
    def onboarding_message(self):
        return " During the onboarding phase, your response was: {} ".format(self.opinion)

    def save_onboarding(self, opinion, threshold):
        if opinion:
            self.evaluate_match(opinion, threshold)
        self.log_agent_opinion(opinion)

    def evaluate_match(self, opinion, threshold):
        loss = cross_entropy(self.label, opinion)
        self.opinion = opinion
        self.is_gold = True if loss < threshold else False
        
    def log_agent_opinion(self, opinion):
        # TODO add to agent
        logging.info("\n".join([
            f"Opinion     : {opinion}",
            f"Gold Agent  : {self.is_gold}",
            "="*80
        ]))

    def add_onboarding(self):
        self.agent = ConversableAgent(
                name=self.agent.name,
                system_message=self.agent.system_message +self.onboarding_message,
                llm_config=self.agent.llm_config
        )
