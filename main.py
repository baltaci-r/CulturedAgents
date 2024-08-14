import json, os.path
from argparse import ArgumentParser
from datasets import load_dataset

from chat import Chat
from utils import * 
from builders import *


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--config_file_or_env', type=str)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--filter', type=str, choices=['region', 'opinion'], default='opinion')
    parser.add_argument('--mode', type=str, choices=['debate', 'collab'], default='collab')
    args = parser.parse_args()

    templates = Templates()
    examples = load_dataset("Anthropic/llm_global_opinions", split='train')
    examples = examples.filter(validate_agents_count)
     
    args.group_size = 5 
    entropy_counts = {0: 0, 0.72:0, 0.97:0, 1.37: 0, 1.52: 0, 1.92: 0, 2.32: 0}

    for ind, example in enumerate(examples):
        save_path = f"{args.save_dir}/{ind}.json"
        if not os.path.exists(save_path):
            print(save_path)
            example = Example(example)
            prompts = Prompts(templates, example, args.mode)
            chat = Chat(args)

            chat.create_agents(example.names, example.labels, example.system_messages) # Add to self.agents
            chat.onboard_agents(example.chars, prompts.onboarding_question)
            
            if len(chat.gold_agents) >= args.group_size:
                entropy = chat.select_agents(entropy_counts, args.filter)
                chat.create_onboarded_agents()
                try: 
                    chat.group_chat(example.chars, prompts.discussion_question + prompts.termination_notice, 
                        prompts.group_opinion_summary_args, prompts.agent_reply_summary_prompt
                    )
                    chat.reflect_agents(example.chars, prompts.reflection_question)
                except Exception as e: 
                    chat.add_exception(e)

                with open(save_path, "w") as f:
                    json.dump(chat.results_json(example.data, entropy), f, indent=2)
