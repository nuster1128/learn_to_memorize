from openai import OpenAI
from langchain.prompts import PromptTemplate
from memengine import FUMemory, LTMemory, STMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY, DEFAULT_LTMEMORY, DEFAULT_STMEMORY
from memengine.utils.AutoSelector import generate_candidate, automatic_select
from memengine.config.Config import MemoryConfig

# ----- Configuration for Dialogue Environment and Agent -----

OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_APIKEY = 'sk-02otYN1Q5IcaC3kewSHonCROpAJCcdRSJ8u7mlvjLT6GY1Ut'
OPENAI_APIBASE = 'https://api.chatanywhere.tech/v1'
MAX_STEP = 5

DialogueAgentPrompt = """Please play the role of a/an {role}.
Your memory is as follows:
{memory_context}
Your current observation is as follows:
{observation}
Please generate a response towards the current observation into one sentence.
{role}: """

EvalPrompt = """You are an experienced dialogue quality evaluator. I will provide you with a dialogue between a user and an assistant.
Please assess the quality of the agent's responses, with a focus on aspects such as fluency, logic, and user satisfaction.
On the scale of 1 to 10, where 1 is the lowest quality and 10 is highest quality.
[Dialogue]
{dialogue}

Your should just output the rating number between from 1 to 10, and do not output any other texts."""

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
    def __init__(self, role, another_role, memory):
        self.llm = LLM()

        self.role = role
        self.another_role = another_role
        self.memory = memory
    
    def response(self, observation):
        prompt = PromptTemplate(
                input_variables=['role', 'memory_context', 'observation'],
                template= DialogueAgentPrompt,
            ).format(role = self.role, memory_context = self.memory.recall(observation), observation = observation)
        res = self.llm.fast_run(prompt)
        self.memory.store('%s: %s\n%s: %s' % (self.another_role, observation, self.role, res))
        return res

# -----  Dialogue Quality Evaluation for Assiatant Agent -----

def eval_assistant(dialogue_record):
    llm = LLM()
    prompt = PromptTemplate(
            input_variables=['dialogue'],
            template= EvalPrompt,
        ).format(dialogue = '\n'.join(dialogue_record))
    res = llm.fast_run(prompt)
    return float(eval(res))/10.0

# ----- Prepare the Reward Function -----
# Here, we utilize the performance of agents in dialogue tasks to reflect the performance of memory.

def sample_reward_func(memory):
    """Given a memory, utilize it and obtain a reward score to reflect how good it is.

    Args:
        memory (BaseMemory): the memory in MemEngine.

    Returns:
        float: the reward score to reflect how good the memory is.

    """
    dialogue_record = []

    user = DialogueAgent('User', 'Assistant', FUMemory(MemoryConfig(DEFAULT_FUMEMORY)))
    assistant = DialogueAgent('Assistant', 'User', memory)
    assistant_response = assistant.response('Please start the dialogue between User and Assistant.')

    for current_step in range(MAX_STEP):
        user_response = user.response(assistant_response)
        assistant_response = assistant.response(user_response)
        dialogue_record.append('User: %s' % user_response)
        dialogue_record.append('Assistant: %s' % assistant_response)

    score = eval_assistant(dialogue_record)
    return score

# ----- Prepare the Range of Model/Config Selection -----

# Option 1: Direct Assign
ModelCandidate = [{
    'model': 'FUMemory',
    'config': DEFAULT_FUMEMORY
},  {
    'model': 'LTMemory',
    'config': DEFAULT_LTMEMORY
},  {
    'model': 'STMemory',
    'config': DEFAULT_STMEMORY
}]

# Option 2: Generate with Combination (Recommended for Hyper-parameter Tuning)
ModelCandidate += generate_candidate({
    'model': 'LTMemory',
    'base_config': DEFAULT_LTMEMORY,
    'adjust_name': 'recall.text_retrieval.topk',
    'adjust_range': [1, 3, 5, 10]
})

# ----- Start Automatic Selection -----

def sample_automode():
    selection_result = automatic_select(sample_reward_func, ModelCandidate)
    print('The full ranking of candidate is shown as follows:')
    print(selection_result)

    print('The best model/config is shown as follows:')
    print(selection_result[0])

if __name__ == "__main__":
    sample_automode()