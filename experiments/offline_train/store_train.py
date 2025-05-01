import os, requests, json
from openai import OpenAI

LocalModelList = ['Qwen2.5-7B']

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class LLM():
    def __init__(self, llm_config):
        self.llm_config = llm_config

        if llm_config['name'] not in LocalModelList:
            self.client = OpenAI(api_key=self.llm_config['api_key'], base_url=self.llm_config['api_base'])

    def parse_response(self, response):
        return {'result': response.choices[0].message.content}

    def request(self, route_path, data):
        post_path = os.path.join(self.llm_config['api_base'], route_path)
        response = requests.post(post_path, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            return False

    def run(self, message_list):
        if self.llm_config['name'] not in LocalModelList:
            response = self.client.chat.completions.create(
                model=self.llm_config['name'],
                messages=message_list,
                temperature=0.9
            )
            response = self.parse_response(response)
        else:
            res = self.request('inference/', data={
            'messages': message_list,
            'kwargs': {}
            })
            if not res:
                response = {'result': False}
            else:
                response = {'result': res['response']}
            # print(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        max_retry = 5
        while not response['result']:
            max_retry -= 1
            response = self.run([{"role": "user", "content": query}])
            print('LLM Inference Retry.')
            if max_retry == 0:
                return 'None'
        return response['result']

llm_config = {
    'name': 'Qwen2.5-7B',
    'api_base': '[Server]'
}

positive_prompt = """You are an expert who is good at reflection.
I am extracting key information from the raw observation to assist in answering the following question.
Here is one of my successful cases, please help me reflect and summarize the successful experience of the extraction process.

Raw Observation:
{raw_observation}

Extracted Key Information:
{extracted_memory}

Question: {question}
Answer: {answer}

Requirements:
1. The summarized successful experience should be general, rather than containing specific details of this question.
2. The summarized successful experience should be useful and helpful for future information extraction.
3. The summarized successful experience should be concise, usually consisting of one or two sentences.
4. Please only output a summary of successful experience in one line, without providing a thought process or explanation."""

negative_prompt = """You are an expert who is good at reflection.
I am extracting key information from the raw observation to assist in answering the following question.
Here is one of my failed cases, please help me reflect and summarize the failure experience of the extraction process.

Raw Observation:
{raw_observation}

Extracted Key Information:
{extracted_memory}

Question: {question}
Answer: {answer}

Requirements:
1. The summarized failure experience should be general, rather than containing specific details of this question.
2. The summarized failure experience should be useful and helpful for future information extraction.
3. The summarized failure experience should be concise, usually consisting of one or two sentences.
4. Please only output a summary of failure experience in one line, without providing a thought process or explanation."""

summarize_prompt = """You are an expert at summarizing, please help me summarize the following experience into a paragraph.

Experience:
{current_context}
{experience_chunk}

Requirements:
1. The summarized experience should be useful and helpful for future information extraction.
2. Please only output the summary in one line, without providing a thought process or explanation.
3. The summarized experience should be rich, comprehensive, and informative."""

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                json_obj = json.loads(stripped_line)
                data.append(json_obj)
    return data

def store_offline_train(experience_path, hint_path, chunk_size):
    # chunk_size = 10
    llm = LLM(llm_config)

    store_positive_path = 'store_positive.json'
    store_negative_path = 'store_negative.json'

    if not os.path.exists(experience_path):
        postive_data = load_json(store_positive_path)
        negative_data = load_json(store_negative_path)

        for pd_list in postive_data:
            for pd in pd_list:
                prompt = positive_prompt.format(**pd)
                print(prompt)
                response = llm.fast_run(prompt)
                print(response)
                with open(experience_path, 'a', encoding='utf-8') as f:
                    json_line = json.dumps({
                        'experience': response
                    },ensure_ascii=False)
                    f.write(json_line + '\n')

        for pd_list in negative_data:
            for pd in pd_list:
                prompt = positive_prompt.format(**pd)
                print(prompt)
                response = llm.fast_run(prompt)
                print(response)
                with open(experience_path, 'a', encoding='utf-8') as f:
                    json_line = json.dumps({
                        'experience': response
                    },ensure_ascii=False)
                    f.write(json_line + '\n')

    exp_chunk = []
    current_context = 'Summarize carefully.'
    experience_list = read_jsonl(experience_path)
    for experience in experience_list:
        exp_chunk.append(experience['experience'])
        if len(exp_chunk) == chunk_size:
            prompt = summarize_prompt.format(**{
                'experience_chunk': '\n'.join(exp_chunk),
                'current_context': current_context
            })
            print(prompt)
            current_context = llm.fast_run(prompt)
            print(current_context)
            exp_chunk = []
        
    with open(hint_path,'w', encoding='utf-8') as f:
        json.dump({
            'hint': current_context
        },f, indent=4,ensure_ascii=False)

if __name__ == '__main__':
    chunk_size = 10
    experience_path = 'experience.jsonl'
    hint_path = '[Path]'

    store_offline_train(experience_path, hint_path, chunk_size)