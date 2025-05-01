from MemEngine.memengine.memory.BaseMemory import ExplicitMemory
from MemEngine.memengine.operation.Recall import BaseRecall
from MemEngine.memengine.operation.Store import BaseStore
from MemEngine.memengine.operation.Optimize import BaseOptimize
from MemEngine.memengine.function.Truncation import LMTruncation
from MemEngine.memengine.function.Utilization import BaseUtilization
from MemEngine.memengine.function.Retrieval import TimeRetrieval, TextRetrieval, ValueRetrieval
from MemEngine.memengine.function.Summarizer import LLMSummarizer
from MemEngine.memengine.utils.Storage import LinearStorage
from MemEngine.memengine.utils.Display import ScreenDisplay
from MemEngine.memengine.operation.Recall import __recall_convert_str_to_observation__
from MemEngine.memengine.operation.Store import __store_convert_str_to_observation__

from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import os
from peft import PeftModel
import numpy as np
import random

random.seed(1128)
np.random.seed(1128)

class MoEGate(nn.Module):
    def __init__(self, moe_mode, embedding_size, hidden_size,  output_size):
        super().__init__()

        self.query_hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.memory_hidden_layer = nn.Linear(embedding_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        if moe_mode == 'general':
            raise
        elif moe_mode == 'trainable':
            self.initialize_trainable_mode()
        else:
            raise "Mode Error!"
    
    def initialize_trainable_mode(self):
        self.initialize_trainable_layer(self.query_hidden_layer)
        self.initialize_trainable_layer(self.memory_hidden_layer)
        self.initialize_trainable_layer(self.output_layer)
    
    def initialize_trainable_layer(self, layer):
        torch.nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        torch.nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, query, memory):
        # print(query.shape,memory.shape)

        query_h = self.query_hidden_layer(query)
        memory_h = self.memory_hidden_layer(memory)
        # print(query_h.shape, memory_h.shape)

        h = torch.sigmoid(query_h + memory_h)
        return torch.softmax(self.output_layer(h),dim=1)

class ImportanceScorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_size = config['embedding_size']

        self.W_q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_m = nn.Linear(self.embedding_size, self.embedding_size)
    
    def forward(self, h_q, h_m):
        e_q = self.W_q(h_q)
        e_m = self.W_m(h_m)

        score = torch.cosine_similarity(e_q,e_m)
        return score

class EmotionScorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_category']

        self.hidden_layer = nn.Linear(self.embedding_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 8)
    
    def forward(self, x):
        h = torch.tanh(self.hidden_layer(x))

        score = self.output_layer(h)
        return score


class ScoreModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.embedding_size = config.embedding_size
        self.time_rank = config.time_rank
        self.output_size = self.time_rank + 3
        self.moe_gate = MoEGate(config.moe_mode, config.embedding_size, config.hidden_size, self.output_size)
        self.moe_gate.cuda()

        self.initialize_metrics(config.metrics_path)
        if hasattr(config, 'params_path'):
            self.initialize_params(config.params_path)
    
    def initialize_metrics(self, metrics_path):
        if metrics_path == False:
            raise
        else:
            importance_param_path = os.path.join(metrics_path, 'importance_score.pickle')
            self.importance_scorer = ImportanceScorer({
                'embedding_size': self.embedding_size
            })

            self.importance_scorer.load_state_dict(torch.load(importance_param_path).state_dict())
            self.importance_scorer = self.importance_scorer.cuda()
            
            emotion_param_path = os.path.join(metrics_path,'emotion_score.pickle')
            self.emotion_scorer = EmotionScorer({
                'embedding_size': self.embedding_size,
                'hidden_size': 256,
                'output_category': 8,
            })
            self.emotion_scorer.load_state_dict(torch.load(emotion_param_path).state_dict())
            self.emotion_scorer = self.emotion_scorer.cuda()

    def initialize_params(self, params_path):
        self.moe_gate.load_state_dict(torch.load(params_path).state_dict())
        self.moe_gate = self.moe_gate.cuda()

    def calculate_recency_scores(self, query_time, memory_time, normal_time):
        delta_time = (query_time - memory_time)/normal_time
        recency_scores = torch.ones(delta_time.size()).unsqueeze(dim=0).cuda()

        for i in range(self.time_rank-1):
            recency_scores = torch.cat((recency_scores, (recency_scores[-1] * delta_time).unsqueeze(dim=0)), dim=0)

        return recency_scores.t()
    
    def calculate_emotion_scores(self, query, memory):
        query_emo = self.emotion_scorer(query)
        query_emo_norm = torch.norm(query_emo, p=2, dim=1)
        memory_emo = self.emotion_scorer(memory)
        memory_emo_norm = torch.norm(memory_emo, p=2, dim=1)

        emotion_scores = torch.matmul(memory_emo, query_emo.squeeze()) / query_emo_norm / memory_emo_norm
        return emotion_scores
    
    def calculate_importance_scores(self, query, memory):
        original_importance_score = self.importance_scorer(query, memory)
        scaled_importance_score = torch.sigmoid(original_importance_score)
        return scaled_importance_score
    
    def forward(self, query, memory, memory_time, query_time, normal_time):
        moe_gate_score = self.moe_gate(query, memory)
        semantic_score = torch.matmul(memory, query.squeeze())
        recency_scores = - 1 * self.calculate_recency_scores(query_time, memory_time, normal_time)
        emotion_score = self.calculate_emotion_scores(query, memory)
        importance_score = self.calculate_importance_scores(query, memory)
        scores = torch.cat((semantic_score.unsqueeze(dim=1), recency_scores, emotion_score.unsqueeze(dim=1), importance_score.unsqueeze(dim=1)),dim=1)
        combined_score = torch.mul(moe_gate_score, scores).sum(dim=1)

        return combined_score

# ----- Utilize Qwen-2.5-7B as Aggregator in Memory Utilization -----

class QwenLLM():
    def __init__(self, model_path, sft_model_path=None, dpo_model_path=None, online_peft_paths=[]):
        self.model = self.create_model(model_path, sft_model_path, dpo_model_path, online_peft_paths)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def create_model(self, model_path, sft_model_path, dpo_model_path, online_peft_paths):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if sft_model_path:
            model = PeftModel.from_pretrained(model, model_id=sft_model_path)
        if dpo_model_path:
            model = PeftModel.from_pretrained(model, model_id=dpo_model_path)
        
        for model_path in online_peft_paths:
            model = PeftModel.from_pretrained(model, model_id=model_path)

        return model
    
    def inference(self, messages, kwargs=None):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=8192).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def fast_run(self, prompt):
        return self.inference([
            {"role": "user", "content": prompt},
        ])

# ----- Memory Utilization -----

class RUSUtilization(BaseUtilization):
    def __init__(self, config, truncation):
        super().__init__(config)

        self.truncation = truncation
        if hasattr(self.config.aggregator_config, 'online_model_path_list'):
            self.aggregate_llm = QwenLLM(
                self.config.aggregator_config.model_path,
                self.config.aggregator_config.sft_model_path,
                self.config.aggregator_config.dpo_model_path,
                self.config.aggregator_config.online_model_path_list
            )
        elif hasattr(self.config.aggregator_config, 'dpo_model_path'):
            self.aggregate_llm = QwenLLM(
                self.config.aggregator_config.model_path,
                self.config.aggregator_config.sft_model_path,
                self.config.aggregator_config.dpo_model_path
            )
        elif hasattr(self.config.aggregator_config, 'sft_model_path'):
            self.aggregate_llm = QwenLLM(
                self.config.aggregator_config.model_path,
                self.config.aggregator_config.sft_model_path
            )
        else:
            self.aggregate_llm = QwenLLM(self.config.aggregator_config.model_path)

    def __aggregate__(self, observation, memory_context, new_memory):
        if memory_context == '':
            new_memory_context = new_memory
        else:
            prompt = PromptTemplate(
                input_variables=self.config.aggregator_config.prompt.input_variables,
                template=self.config.aggregator_config.prompt.template
            ).format(**{
                    'observation': observation,
                    'memory_context': memory_context,
                    'new_memory': new_memory
                })
            # print(prompt)
            new_memory_context = self.aggregate_llm.fast_run(prompt)
        
        print('[Once Aggregate::] New Memory Context::', new_memory_context)
        return new_memory_context

    def __call__(self, observation, ranked_memory_index, storage):
        memory_context = ''
        intermediate_list = []
        previous_length_list = [0]
        c_previous = 1.0
        for current_idx, mid in enumerate(ranked_memory_index):
            new_memory_context = self.__aggregate__(observation, memory_context, storage.get_memory_text_by_mid(mid))
            new_memory_length = self.truncation.get_piece_number(new_memory_context)
            if len(previous_length_list) <= 2:
                previous_length_list.append(new_memory_length)
            else:
                previous_length_list.append(new_memory_length)
                previous_length_list = previous_length_list[1:]
                delta_1, delta_2 = previous_length_list[1] - previous_length_list[0], previous_length_list[2] - previous_length_list[1]
                c_current = np.clip(1.0 * delta_2/abs(delta_1+0.0001), 0.0, 1.0)
                p = max(c_previous, c_current)
                # print(previous_length_list)
                print('[Probability of AGG]::',c_previous, c_current, p)
                c_previous = c_current
                if random.random() > p:
                    return memory_context, intermediate_list
                
            if self.truncation.check_truncation_needed(new_memory_context):
                return memory_context
            memory_context = new_memory_context
            intermediate_list.append(new_memory_context)

        print('[Complete Once Utilization Operation]::', memory_context)
        return memory_context, intermediate_list

# ----- Memory Recall -----

class StoreCache():
    def __init__(self):
        self.cache_text = []
        self.cache_time = None
    
    def add(self, text, timestamp):
        self.cache_text.append(text)
        self.cache_time = timestamp
    
    def clear(self):
        self.cache_text = []
        self.cache_time = None
    
    def get_cache(self):
        return '\n'.join(self.cache_text), self.cache_time

class RUSMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)
        self.storecache = StoreCache()

        self.score_model = ScoreModel(self.config.score_config)

        self.truncation = LMTruncation(self.config.truncation)
        self.utilization = RUSUtilization(self.config.utilization, self.truncation)

        self.summarizer_with_hints = eval(self.config.summarizer_with_hints.method)(self.config.summarizer_with_hints)
        self.summarizer_without_hints = eval(self.config.summarizer_without_hints.method)(self.config.summarizer_without_hints)
        self.hint = {'hint': self.config.initial_hint}
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.time_retrieval])

    def __extract__(self, text):
        if self.config.hint_usage and self.hint != '':
            text = self.truncation(text)
            return self.summarizer_with_hints({'observation': text, 'hint': self.hint['hint']})
        else:
            return self.summarizer_without_hints({'observation': text})

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        # ---- Dump Cache ----
        cache_text, cache_time = self.storecache.get_cache()
        if len(cache_text) != 0:
            memory_info = self.__extract__(cache_text)

            self.storage.add({'time': cache_time, 'text': memory_info})
            self.text_retrieval.add(memory_info)
            self.time_retrieval.add(cache_time)
            print('[Complete Once Store Operation (To Storage)]::', memory_info)

        self.storecache.clear()
        # ---- End ----

        if self.storage.is_empty():
            return self.config.empty_memory

        text = query['text']
        if 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']
        
        # [Memory Recall Process]
        memory_text_embeddings = self.text_retrieval.tensorstore
        memory_time_embeddings = self.time_retrieval.tensorstore
        query_embedding = self.text_retrieval.encoder(text, return_type='tensor')

        with torch.no_grad():
            score_tensor = self.score_model(query_embedding, memory_text_embeddings, memory_time_embeddings, torch.tensor(timestamp).cuda(), torch.tensor(timestamp).cuda()-memory_time_embeddings[0])

        # print('Score Tensor Shape:', score_tensor)
        # print(score_tensor)

        sorted_score, indices = torch.sort(score_tensor, descending=True)
        ranked_memory_index = indices.cpu().numpy()
        # print('[RankedMemoryIndex::]', ranked_memory_index)
        # print(ranked_memory_index)

        # [Memory Utilization Process]
        memory_context, intermediate_list = self.utilization(text, ranked_memory_index, self.storage)
        print('[Complete Once Recall Operation]::', memory_context)

        return self.truncation(memory_context), {
            'retrieve': {
                'type': 'retrieval',
                'query_text': text,
                'memory_text': [item['text'] for item in self.storage.memory_list],
                'memory_time': memory_time_embeddings.cpu().numpy().tolist(),
                'current_time': timestamp,
                'normal_time': timestamp - memory_time_embeddings.cpu().numpy().tolist()[0],
                'ranked_memory_index': ranked_memory_index.tolist()
            },
            'utilize': {
                'type': 'utilization',
                'intermediate_list': intermediate_list
            },
            'store': {
                'type': 'store',
                'raw_observation': cache_text,
                'extracted_memory': memory_info
            }
        }

# ----- Memory Store -----

class RUSMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.time_retrieval = kwargs['time_retrieval']
        self.storecache = kwargs['storecache']
    
    def reset(self):
        pass
        # self.hint = {'hint': self.config.initial_hint}

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.storage.counter
        else:
            timestamp = observation['time']
        text = observation['text']

        # memory_info = self.__extract__(text)
        # observation['text'] = memory_info

        # self.storage.add(observation)
        # self.text_retrieval.add(memory_info)
        # self.time_retrieval.add(timestamp)
        self.storecache.add(text, timestamp)
        print('[Complete Once Store Operation (To Cache)]')


# ----- Memory Optimization -----

class RUSOptimize(BaseOptimize):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        # self.reflector = TrialReflector(config.reflector)
        # self.insight = kwargs['insight']
        raise
    
    def reset(self):
        self.__reset_objects__([self.reflector])

    def __call__(self, **kwargs):
        new_trial = kwargs['new_trial']

        new_insight = self.reflector.generate_insight({
            'previous_insight': self.insight['global_insight'],
            'new_trial': new_trial,
            'example': self.config.reflector.example
        })

        self.insight['global_insight'] = new_insight


# ----- RUSMemory -----

class RUSMemory(ExplicitMemory):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = LinearStorage(self.config.args.storage)

        self.recall_op = RUSMemoryRecall(
            self.config.args.recall,
            storage = self.storage
        )
        self.store_op = RUSMemoryStore(
            self.config.args.store,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval,
            storecache = self.recall_op.storecache
        )

        self.auto_display = ScreenDisplay(self.config.args.display, register_dict = {
            'Memory Storage': self.storage,
            'hint': self.recall_op.hint
        })
        

        # self.optimize_op = RFOptimize(self.config.args.optimize, insight = self.insight)

    def reset(self):
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
    
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        # self.optimize_op(**kwargs)
        raise