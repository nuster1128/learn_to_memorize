import logging, json, random, string, re
from libzim.reader import Archive
from libzim.search import Query, Searcher
from bs4 import BeautifulSoup

class WikiSearcher():
    def __init__(self, wiki_path):
        self.zim = Archive(wiki_path)
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
    def __init__(self, environment_config):
        self.environment_config = environment_config
        self.test_level = self.environment_config['level']
        self.dataset = self.load_dataset()
        self.wiki_searcher = WikiSearcher(self.environment_config['wiki_path'])
        self.current_train_id = 0
        self.current_test_id = 0
    
    def load_json(self, path):
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_dataset(self):
        # 99447 in the training dataset; 7405 in the testing dataset.
        train_path = '%s/%s' % (self.environment_config['dataset_path'], 'hotpot_train_v1.1_10000.json')
        train_data = self.load_json(train_path)
        test_path = '%s/%s' % (self.environment_config['dataset_path'], f'hotpot_test_v1_{self.test_level}.json')
        test_data = self.load_json(test_path)

        return {'train_data': train_data, 'test_data': test_data}

    def reset_train(self):
        self.current_try = 0
        current_task = self.dataset['train_data'][self.current_train_id]
        self.task_info = {
            '_id': current_task['_id'],
            'question': current_task['question'],
            'answer': current_task['answer'],
            'type': current_task['type'],
            'level': current_task['level']
        }

        logging.info("[Training Reset] Task has been reset successully: %s." % self.task_info)
        self.current_train_id += 1
        return {'question':self.task_info['question']}, 0, False, {'signal':'Success.', 'step_id': self.current_try}

    def reset_test(self):
        self.current_try = 0
        current_task = self.dataset['test_data'][self.current_test_id]
        self.task_info = {
            '_id': current_task['_id'],
            'question': current_task['question'],
            'answer': current_task['answer'],
            'type': current_task['type'],
            'level': current_task['level']
        }

        logging.info("[Testing Reset] Task has been reset successully: %s." % self.task_info)
        self.current_test_id += 1
        return {'question':self.task_info['question']}, 0, False, {'signal':'Success.', 'step_id': self.current_try}

    def get_number_of_train_task(self):
        return self.current_train_id, len(self.dataset['train_data'])
    
    def get_number_of_test_task(self):
        return self.current_test_id, len(self.dataset['test_data'])
    
    def shuffle_train_data(self):
        random.shuffle(self.dataset['train_data'])

    def step(self, action):
        self.current_try += 1
        if not action or action['type'] not in ['Search', 'Finish']:
            return {'res':'Action is inavalid.'}, 0, self.is_max_try(), {'signal':'Fail.', 'step_id': self.current_try}
        observation, reward, terminated, info = getattr(self, action['type'], None)(action['args']['keyword'])
        return observation, reward, terminated, info

    def is_max_try(self) -> bool:
        return self.current_try >= self.environment_config['max_try']

    def Search(self, keyword):
        res = self.wiki_searcher.search(keyword)
        return {'res':res}, 0, self.is_max_try(), {'signal':'Success.', 'step_id': self.current_try}

    def Finish(self, keyword):
        correct = EM(self.task_info['answer'], keyword)
        logging.info("[Eval] Comparing [%s](res) with [%s]" % (keyword, self.task_info['answer']))
        if correct:
            return {'res':'The answer is right.'}, 1, True, {'signal':'Success.', 'step_id': self.current_try}
        else:
            return {'res':'The answer is wrong.'}, 0, True, {'signal':'Success.', 'step_id': self.current_try}