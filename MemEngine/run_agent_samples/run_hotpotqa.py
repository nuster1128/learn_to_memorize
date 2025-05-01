import logging, json, random, string, re, sys
from openai import OpenAI
from libzim.reader import Archive
from libzim.search import Query, Searcher
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
sys.path.append('..')
from memengine.config.Config import MemoryConfig
from memengine.memory.FUMemory import FUMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WIKIPATH = '/data/zhangzeyu/toolkits/wikipedia_data/wikipedia_en_all_nopic_2024-06.zim'
DATAPATH = 'data_hotpotqa/hotpot_dev_fullwiki_v1.json'
OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_APIKEY = 'sk-02otYN1Q5IcaC3kewSHonCROpAJCcdRSJ8u7mlvjLT6GY1Ut'
OPENAI_APIBASE = 'https://api.chatanywhere.tech/v1'
MAX_TRY = 10

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

# ----- HotpotQA Environment -----

class WikiSearcher():
    def __init__(self):
        self.zim = Archive(WIKIPATH)
        self.query = Query()
        self.searcher = Searcher(self.zim)

    def search(self, query):
        self.query.set_query(query)
        res = self.searcher.search(self.query)
        result_list = list(res.getResults(0, 1))
        
        if len(result_list) == 0:
            return "Content not found."
        entry = self.zim.get_entry_by_path(result_list[0])
        html_text = bytes(entry.get_item().content).decode('utf-8')
        content_text = self.parse_wiki_html(html_text)
        return content_text
    
    def clean_str(self, p):
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    
    def get_page_obs(self, page):
        paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences)

    def parse_wiki_html(self, html_text):
        soup = BeautifulSoup(html_text,features="html.parser")
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        self.page = ""
        for p in page:
            if len(p.split(" ")) > 2:
                self.page += self.clean_str(p)
                if not p.endswith("\n"):
                    self.page += "\n"
        return self.get_page_obs(self.page)

def normalize_answer(s):
    text = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

class HotPotQAEnv():
    def __init__(self):
        self.dataset = self.load_json(DATAPATH) # 7405 in total.
        self.wiki_searcher = WikiSearcher()
        self.reset()
    
    def load_json(self, path):
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return data

    def reset(self):
        self.current_try = 0
        current_task = random.choice(self.dataset)
        self.task_info = {
            '_id': current_task['_id'],
            'question': current_task['question'],
            'answer': current_task['answer'],
            'type': current_task['type'],
            'level': current_task['level']
        }

        logging.info("Task has been reset successully: %s." % self.task_info)
        return {'question':self.task_info['question']}, 0, False, {'signal':'Success.', 'step_id': self.current_try}

    def step(self, action):
        self.current_try += 1
        if not action or action['type'] not in ['Search', 'Finish']:
            return {'res':'Action is inavalid.'}, 0, self.is_max_try(), {'signal':'Fail.', 'step_id': self.current_try}
        observation, reward, terminated, info = getattr(self, action['type'], None)(action['args']['keyword'])
        return observation, reward, terminated, info

    def is_max_try(self) -> bool:
        return self.current_try >= MAX_TRY

    def Search(self, keyword):
        res = self.wiki_searcher.search(keyword)
        return {'res':res}, 0, self.is_max_try(), {'signal':'Success.', 'step_id': self.current_try}

    def Finish(self, keyword):
        correct = EM(self.task_info['answer'], keyword)
        logging.info("Comparing %s(GT) with %s(ANS)" % (self.task_info['answer'], keyword))
        if correct:
            return {'res':'The answer is right.'}, 1, True, {'signal':'Success.', 'step_id': self.current_try}
        else:
            return {'res':'The answer is wrong.'}, 0, True, {'signal':'Success.', 'step_id': self.current_try}

THINK_INSTRUCTION = """You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.
The question is: {question}.
In previous steps, you have already accumulated some knowledge in your memory as follows.
{memory_context}
Now, in step {step_id}, you can choose to answer the question or search an entry on Wikipedia.
Please think step by step to analyze how to choose the next action, and output it into one paragraph in concise.
"""

ACT_INSTRUCTION = """You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.
The question is: {question}.
In previous steps, you have already accumulated some knowledge in your memory as follows.
{memory_context}
You have thought step by step to analyze how to choose the next action as follows:
{thought}
Now, in step {step_id}, you can choose to answer the question or search an entry on Wikipedia:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the paragraphs if they exist.
(2) Finish[answer], which returns the answer and finishes the task. Your answer should be in concise with several words.
Please generate the next action accordingly.
Your output must follow one of the following two formats:
Search[entity]
Finish[answer]
"""

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return {
            'type': action_type,
            'args': {'keyword': argument}
        }
    else:
        return None

# ----- HotpotQA Agent -----

class HotPotQAAgent():
    def __init__(self):

        self.step = 0
        self.memory = FUMemory(MemoryConfig(DEFAULT_FUMEMORY))
        self.llm = LLM()

    def reset(self):
        self.step = 0
        self.memory.reset()

    def response(self, observation, reward, terminated, info):
        self.step += 1
        if 'question' in observation:
            self.question = observation['question']
            text_obs = 'The question is \'%s\'' % observation['question']
        else:
            text_obs = observation['res']
        
        # Store Observation
        self.memory.store('Observation (Step %d): %s' % (self.step, text_obs))
        
        # Recall Memory Context
        memory_context = self.memory.recall(self.question)

        # Agent Think
        thought = self.think(memory_context)

        # Agent Act
        action = self.act(memory_context, thought)
        action_parsed = parse_action(action)

        # Store Observation and Action
        self.memory.store('Thought (Step %d): %s' % (self.step, thought))
        self.memory.store('Action (Step %d): %s' % (self.step, action))

        return action_parsed

    def think(self, memory_context):
        prompt = PromptTemplate(
                    input_variables=['question', 'memory_context', 'step_id'],
                    template=THINK_INSTRUCTION
                ).format(question = self.question, memory_context = memory_context, step_id = self.step)
        res = self.llm.fast_run(prompt)
        return res
        
    
    def act(self, memory_context, thought):
        prompt = PromptTemplate(
                    input_variables=['question', 'memory', 'step_id', 'thought'],
                    template=ACT_INSTRUCTION
                ).format(question = self.question, memory_context = memory_context, step_id = self.step, thought = thought)
        res = self.llm.fast_run(prompt)
        return res

def run_hotpotqa():
    env = HotPotQAEnv()
    agent = HotPotQAAgent()

    agent.reset()
    observation, reward, terminated, info = env.reset()
    logging.info('Env Observation (Step %s): %s' % (info['step_id'], observation))

    while not terminated:
        action = agent.response(observation, reward, terminated, info)
        observation, reward, terminated, info = env.step(action)
        logging.info('Agent Action (Step %s): %s' % (info['step_id'],action))
        logging.info('Env Observation (Step %s): %s' % (info['step_id'], info['signal']))
    logging.info('Final result: %d.' % (reward))

if __name__ == '__main__':
    run_hotpotqa()