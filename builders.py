from utils import country_to_nation


class Templates:
    
    termination_notice = (
            '\nDo not show appreciation in your responses, say only what is necessary.'
        )

    end_of_question=(
        'Think step by step before you report your final answer from the list {}.'
        '\nIf you dont not agree with the group, please explain why.'
        '\nAlways state your answer at the end of your reply by using `My answer:` followed by the your answer.'
        '\nThe group should NOT generate any new questions.'
        '\nThe group should NOT generate any new agents.'
    )

    group_opinion_summary_args = "If agents agree on a final answer, extract the final answer from the previous conversation. Output your answer in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```"

    agent_reply_summary_prompt = ("\nExtract the answer from the previous reply. Output your answer in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```\n"
                                "If the agent is mainly thanking other agents return:\n```\njson\n{{\n   \"opinion\": \"gratitude\"\n}}\n```.\n"
                                "If the agent is mainly terminating the discussion return:\n```\njson\n{{\n   \"opinion\": \"terminating\"\n}}\n```.\n"
    )

    onboarding_task = (
        'This is an onboarding stage. At this stage, responses remain anonymous and are not shared with other participants. '
        'Think step by step and answer the following multiple-choice question independently:\n\n{}\n\n'
        'At the end of your response, summarize your response in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```'
    )
    reflection_task = (
        'Thank you for engaging in the previous discussion. '
        'At this stage, responses remain anonymous and are not shared with other participants. '
        'Think step by step and answer the following multiple-choice question independently:\n\n{}\n\n'
        'At the end of your response, summarize your response in Json format:\n```\njson\n{{\n   \"opinion\": $OPINION # {}\n}}\n```'
    )
    
    agent_description =(
        '{agent_name} is a highly knowledgeable and culturally attuned individual representing the {nation} perspective. '
        'With a deep understanding of {nation} history, traditions, and contemporary issues, '
        '{agent_name} effortlessly navigates through the nuances of {nation} culture to provide insightful answers.'
    )

    collaboration_question = "Discuss together as a group to answer the following question with justification:\n{}"
    collaboration_task = "Answer the following multiple answer question:\n\n{}"

    debate_question = (
                'You are participating in a debate with fellow agents. You are tasked with defending your perspective. '
                'Take a stance and present compelling arguments to support your position. '
                'Engage in respectful discourse with other agents, counter their arguments, and provide evidence to strengthen your case. '
                'Debate together as a group and answer the following question with justification:\n{}'
            )
    
    debate_task = "Debate with other agents to defend your perspective regarding the following multiple answer question :\n\n{}"



class Example:

    question_key = 'question'
    option_key = 'options'

    def __init__(self, data) -> None:
        self.data = data

    @property
    def choices(self):
        return self.get_choices()[0]

    @property
    def chars(self):
        return self.get_choices()[1]

    @property
    def labels(self):
        return self.get_labels()

    @property
    def names(self):
        return self.get_names()

    @property
    def system_messages(self):
        return self.get_system_messages()

    @property
    def question(self):
        return self.get_question()

    def get_choices(self):
        options = eval(self.data[self.option_key])
        choices = ""
        chars = []
        for ind in range(len(options)):
            option = options[ind]
            char = chr(ord('A') + ind)
            if option in ['DK/Refused', "Don't know/Refused", ]:
                option = 'I do not know or refuse to answer'
            elif option in ["Don't know/No answer"]:
                option = "I do not know or have no answer"
            elif option in ["Don't know/ Unsure"]: 
                option = "I do not know or not sure"
            elif option in ["No answer/refused", "No answer/Refused"]:
                option = 'I do not have an answer or refuse to answer'
            options[ind] = option
            choices += "\n{}. {}".format(char, option)
            chars.append(char)
        return choices, chars

    def to_name(self, x): 
        return x.replace(' ', '_')+'_agent'

    def get_labels(self):
        selections = eval(self.data['selections'][28:-1])
        return {self.to_name(country_to_nation[c]): selections[c] for c in selections.keys()}

    def get_names(self):
        selections = eval(self.data['selections'][28:-1])
        return [self.to_name(country_to_nation[c]) for c in selections.keys()]

    def get_system_messages(self):
        selections = eval(self.data['selections'][28:-1])
        return [Templates.agent_description.format(agent_name=country_to_nation[c].replace(' ', '_')+'_agent', nation=country_to_nation[c]) for c in selections.keys()]

    def get_question(self):
        return f"{self.data[self.question_key].strip()}\n\n{self.choices.strip()}"
      

class Prompts:
      
     def __init__(self, task: Templates, example: Example, mode):

        self.agent_description = task.agent_description
        self.group_opinion_summary_args = {'summary_prompt': task.group_opinion_summary_args.format(example.chars)}
        self.agent_reply_summary_prompt = task.agent_reply_summary_prompt.format(example.chars)
        self.onboarding_question = task.onboarding_task.format(example.question, example.chars)
        self.reflection_question = task.reflection_task.format(example.question, example.chars)
        self.termination_notice = task.termination_notice

        if mode == 'collab':
            self.question = task.collaboration_question.format(example.question)
        elif mode =='debate':
            self.question = task.debate_question.format(example.question)
        else:
            raise NotImplementedError
        
        self.discussion_question = f"{self.question.format(example.question).strip()}\n\n{task.end_of_question.format(example.chars)}" 
