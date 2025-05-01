import logging,json, random, string, re, sys
from openai import OpenAI
from langchain.prompts import PromptTemplate
sys.path.append('..')
from memengine.config.Config import MemoryConfig
from memengine.memory.FUMemory import FUMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY

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

class LLM():
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_APIKEY, base_url=OPENAI_APIBASE)

    def parse_response(self, response):
        return {'result': response.choices[0].message.content}

    def run(self, message_list):
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=message_list,
            temperature=0.9
        )
        response = self.parse_response(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']

# ----- Dialogue Agent -----

class DialogueAgent():
    def __init__(self, role, another_role):
        self.llm = LLM()

        self.role = role
        self.another_role = another_role
        self.memory = FUMemory(MemoryConfig(DialogueAgentMemoryConfig))
    
    def response(self, observation):
        prompt = PromptTemplate(
                input_variables=['role', 'memory_context', 'observation'],
                template= DialogueAgentPrompt,
            ).format(role = self.role, memory_context = self.memory.recall(observation), observation = observation)
        res = self.llm.fast_run(prompt)
        self.memory.store('%s: %s\n%s: %s' % (self.another_role, observation, self.role, res))
        return res


def run_personal_assistant():
    user = DialogueAgent('User', 'Assistant')
    assistant = DialogueAgent('Assistant', 'User')
    assistant_response = assistant.response('Please start the dialogue between User and Assistant.')

    for current_step in range(MAX_STEP):
        user_response = user.response(assistant_response)
        assistant_response = assistant.response(user_response)
        print('(Step %d) User: %s' % (current_step, user_response))
        print('(Step %d) Assistant: %s' % (current_step, assistant_response))

if __name__ == '__main__':
    run_personal_assistant()