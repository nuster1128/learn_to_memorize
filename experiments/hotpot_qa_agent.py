import logging, json, random, string, re, sys, os, requests
from openai import OpenAI
from langchain.prompts import PromptTemplate
from MemEngine.memengine import *
from experiments.RUSMemory import RUSMemory
from online_training_utils import store_train, store_data_prepare, retrieval_data_prepare, retrieval_train
from online_training_utils import utilization_data_prepare, utilization_train, save_online_params

LocalModelList = ['Qwen2.5', 'Llama-3.1']

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
            print(response)
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

THINK_INSTRUCTION = """You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.
The question is: {question}.
Now, you can choose to answer the question or search an entity on Wikipedia.
Please think step by step to analyze how to choose the next action, and output it into one paragraph in concise.

In previous steps, you have already accumulated some knowledge in your memory as follows:
{memory_context}
"""

ACT_INSTRUCTION = """You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.
The question is: {question}.
You have thought step by step to analyze how to choose the next action as follows:
{thought}
Now, you can choose to answer the question or search an entry on Wikipedia:
(1) Search[entity], which searches the entity on Wikipedia and returns the paragraphs if they exist.
(2) Finish[answer], which returns the answer and finishes the task. Your answer should be in concise with several words, NOT a sentence.
Please generate the next action accordingly.
Your output must follow one of the following two formats:
Search[entity]
Finish[answer]
Here are some examples:
Search[Alan Turing]
Finish[no]
Finish[Shanghai]

In previous steps, you have already accumulated some knowledge in your memory as follows:
{memory_context}
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
        logging.warning('Parse Action Fail for Agent.')
        return None

class HotPotQAAgent():
    def __init__(self, agent_config):

        self.agent_config = agent_config
        self.llm_config = agent_config['llm_config']
        self.memory_config = agent_config['memory_config']

        self.step = 0
        self.memory = eval(self.memory_config['method'])(MemoryConfig(self.memory_config['config']))
        self.llm = LLM(self.llm_config)

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

        operation_list = []
        
        # Store Observation
        self.memory.store('Observation (Step %d): %s' % (self.step, text_obs))
        
        # Recall Memory Context
        memory_context, memory_info = self.memory.recall(self.question)
        operation_list.append(memory_info['store'])
        operation_list.append(memory_info['retrieve'])
        operation_list.append(memory_info['utilize'])
        logging.info('[agent] [Memory] %s' % memory_context)

        # Agent Think
        thought = self.think(memory_context)
        logging.info('[agent] [Thought] %s' % thought)

        # Agent Act
        action = self.act(memory_context, thought)
        action_parsed = parse_action(action)

        # Store Observation and Action
        self.memory.store('Thought (Step %d): %s' % (self.step, thought))
        self.memory.store('Action (Step %d): %s' % (self.step, action))

        return action_parsed, {
            'step_id': self.step,
            'operation_list': operation_list
        }

    def think(self, memory_context):
        prompt = PromptTemplate(
                    input_variables=['question', 'memory_context'],
                    template=THINK_INSTRUCTION
                ).format(question = self.question, memory_context = memory_context)
        res = self.llm.fast_run(prompt)
        return res
        
    def act(self, memory_context, thought):
        prompt = PromptTemplate(
                    input_variables=['question', 'memory', 'thought'],
                    template=ACT_INSTRUCTION
                ).format(question = self.question, memory_context = memory_context, thought = thought)
        # logging.info('[Action Prompt] %s' % prompt)
        res = self.llm.fast_run(prompt)
        # logging.info('[Action] %s' % res)
        return res

    def online_train(self, environment):

        save_dir = self.memory.config.args.online_train.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        total_epoch = self.memory.config.args.online_train.train_epoch
        sample_batch = self.memory.config.args.online_train.sample_batch

        chunk_size = self.memory.config.args.online_train.chunk_size
        sft_lr = self.memory.config.args.online_train.sft_lr
        sft_gb = self.memory.config.args.online_train.sft_gb
        dpo_lr = self.memory.config.args.online_train.dpo_lr
        dpo_gb = self.memory.config.args.online_train.dpo_gb
        sft_cache_path = f'{save_dir}/sft_cache_path'
        dpo_cache_path = f'{save_dir}/dpo_cache_path'

        if not os.path.exists(sft_cache_path):
            os.mkdir(sft_cache_path)
        
        if not os.path.exists(dpo_cache_path):
            os.mkdir(dpo_cache_path)

        for epoch in range(total_epoch):
            total_score = 0
            print(f'----- [Epoch {epoch}] Start -----')
            # Sample Process
            batch_data = []
            for sample_idx in range(sample_batch):
                print(f'----- Sample {sample_idx} -----')
                self.reset()
                observation, reward, terminated, info = environment.reset_train()
                # logging.info('Environment Observation (%s): %s' % (info['step_id'], observation))

                step_info_list = []
                while not terminated:
                    action, step_info = self.response(observation, reward, terminated, info)
                    step_info_list.append(step_info)
                    # logging.info('Agent Action (%s): %s' % (info['step_id'],action))
                    observation, reward, terminated, info = environment.step(action)
                    # logging.info('Environment Observation (%s): %s' % (info['step_id'], observation))
                logging.info('[Trjectory %d] Final result: %d.' % (sample_idx, reward))
                total_score += reward
                logging.info('[Trjectory %d] Total Score: %d.' % (sample_idx, total_score))
                trial_info = {
                    'task_id': environment.current_train_id,
                    'task_info': environment.task_info,
                    'step_info_list': step_info_list,
                    'total_step': info['step_id'],
                    'reward': reward
                }
                batch_data.append(trial_info)

            print('----- Batch Data -----')
            print(len(batch_data), batch_data)
            print('----- Batch Data End -----')

            # Training Process

            ## Optimize Store
            print('----- Online Store -----')
            positive_data_list, negative_data_list = store_data_prepare(batch_data)
            print('[Positive Data List]::', len(positive_data_list), positive_data_list)
            print('[Negative Data List]::', len(negative_data_list), negative_data_list)
            print('[Old Hint]::', self.memory.recall_op.hint['hint'])
            optimized_hint = store_train(positive_data_list, negative_data_list, self.memory.recall_op.summarizer_with_hints.llm, self.memory.recall_op.hint, chunk_size)
            self.memory.recall_op.hint['hint'] = optimized_hint
            print('[Optimized Hint]::', self.memory.recall_op.hint['hint'])
            print('----- Online Store End -----')

            ## Optimize Retrieval
            print('----- Online Retrieval -----')
            retrieval_tensor = retrieval_data_prepare(batch_data)
            print('[Retrieval Tensor]::', retrieval_tensor)
            if retrieval_tensor[0] is None:
                print('----- No Retrieval Data End -----')
            else:
                retrieval_train(retrieval_tensor, self.memory.recall_op.score_model)
                print('----- Online End -----')

            ## Optimize Utilization
            print('----- Online Utilization -----')
            data_list, expert_data = utilization_data_prepare(batch_data)
            print('[Data List]::', data_list)
            print('[Expert Data]::', expert_data)
            optimized_model = utilization_train(
                self.memory.recall_op.utilization.aggregate_llm.model,
                self.memory.recall_op.utilization.aggregate_llm.tokenizer,
                data_list,
                expert_data,
                epoch,
                sft_cache_path, dpo_cache_path, sft_gb, sft_lr, dpo_gb, dpo_lr
            )
            self.memory.recall_op.utilization.aggregate_llm.model = optimized_model
            print('----- Online Utilization End -----')
        
        print('----- All Online Training End -----')
        save_online_params(
            self.memory.recall_op.hint['hint'],
            self.memory.recall_op.utilization.aggregate_llm.model,
            self.memory.recall_op.score_model.moe_gate,
            save_dir
        )
        print('----- Online Params Save Finish -----')

