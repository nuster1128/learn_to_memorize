import logging,json, random, string, re, sys
from openai import OpenAI
from langchain.prompts import PromptTemplate
sys.path.append('..')
from memengine.config.Config import MemoryConfig
from memengine.memory.FUMemory import FUMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY
from autogen import ConversableAgent

OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_APIKEY = 'sk-02otYN1Q5IcaC3kewSHonCROpAJCcdRSJ8u7mlvjLT6GY1Ut'
OPENAI_APIBASE = 'https://api.chatanywhere.tech/v1'
MAX_STEP = 10

DialogueAgentMemoryConfig = DEFAULT_FUMEMORY
DialogueAgentMemoryConfig['recall']['utilization']['list_config']['index'] = False

DialogueAgentPrompt = """Please play the role of a/an {role}.
Your memory is as follows:
{memory_context}
Your current observation is as follows:
{observation}
Please generate a response towards the current observation into one sentence.
{role}: """

# ----- Dialogue Agent -----

class DialogueAgent():
    def __init__(self, role, another_role):
        self.role = role
        self.another_role = another_role

        self.conversable_agent = ConversableAgent(
            self.role,
            llm_config = {
                'model': OPENAI_MODEL,
                'api_key': OPENAI_APIKEY,
                'base_url': OPENAI_APIBASE
            },
            human_input_mode="NEVER"
        )
        self.memory = FUMemory(MemoryConfig(DialogueAgentMemoryConfig))
    
    def response(self, observation):
        prompt = PromptTemplate(
                input_variables=['role', 'memory_context', 'observation'],
                template= DialogueAgentPrompt,
            ).format(role = self.role, memory_context = self.memory.recall(observation), observation = observation)

        res = self.conversable_agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
        self.memory.store('%s: %s\n%s: %s' % (self.another_role, observation, self.role, res))
        return res

def run_autogen_sample():
    user = DialogueAgent('User', 'Assistant')
    assistant = DialogueAgent('Assistant', 'User')
    assistant_response = assistant.response('Please start the dialogue between User and Assistant.')

    for current_step in range(MAX_STEP):
        user_response = user.response(assistant_response)
        assistant_response = assistant.response(user_response)
        print('(Step %d) User: %s' % (current_step, user_response))
        print('(Step %d) Assistant: %s' % (current_step, assistant_response))

if __name__ == '__main__':
    run_autogen_sample()