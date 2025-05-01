from memengine.function import BaseJudge, LMTruncation, ConcateUtilization, TextRetrieval, ValueRetrieval
from memengine.operation import BaseRecall, BaseStore, RFOptimize
from memengine.operation.Recall import __recall_convert_str_to_observation__
from memengine.operation.Store import __store_convert_str_to_observation__
from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils import LinearStorage, ScreenDisplay
from memengine.config.Config import MemoryConfig
from default_config.DefaultGlobalConfig import DEFAULT_GLOBAL_CONFIG
from default_config.DefaultUtilsConfig import DEFAULT_LINEAR_STORAGE, DEFAULT_SCREEN_DISPLAY
from default_config.DefaultOperationConfig import DEFAULT_RFMEMORY_OPTIMIZE
from default_config.DefaultFunctionConfig import DEFAULT_LMTRUNCATION, DEFAULT_CONCATE_UTILIZATION, DEFAULT_TEXT_RETRIEVAL, DEFAULT_VALUE_RETRIEVAL
import random, torch

# ----- Configuration -----

MyMemoryConfig = {
    'name': 'MyMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': {
        'method': 'MyMemoryRecall',
        'truncation': DEFAULT_LMTRUNCATION,
        'utilization': DEFAULT_CONCATE_UTILIZATION,
        'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
        'bias_retrieval': DEFAULT_VALUE_RETRIEVAL,
        'topk': 3,
        'empty_memory': 'None'
    },
    'store': {
        'method': 'MyMemoryStore',
        'bias_judge': {
            'method': 'MyBiasJudge',
            'scale': 2.0
        }
    },
    'optimize': DEFAULT_RFMEMORY_OPTIMIZE,
    'display': DEFAULT_SCREEN_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

# ----- Customize Memory Functions -----

class MyBiasJudge(BaseJudge):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, text):
        return random.random()/self.config.scale

# ----- Customize Memory Operation -----

class MyMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.insight = kwargs['insight']
        self.truncation = LMTruncation(self.config.truncation)
        self.utilization = ConcateUtilization(self.config.utilization)
        self.text_retrieval = TextRetrieval(self.config.text_retrieval)
        self.bias_retrieval = ValueRetrieval(self.config.bias_retrieval)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.bias_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        
        relevance_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
        bias, _ = self.bias_retrieval(None, topk=False, with_score = True, sort = False)
        final_scores = relevance_scores + bias
        scores, ranking_ids = torch.sort(final_scores, descending=True)

        if hasattr(self.config, 'topk'):
            scores, ranking_ids = scores[:self.config.topk], ranking_ids[:self.config.topk]

        memory_context = self.utilization({
                    'Insight': self.insight['global_insight'],
                    'Memory': [self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids]
                })

        return self.truncation(memory_context)

class MyMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.bias_retrieval = kwargs['bias_retrieval']

        self.bias_judge = MyBiasJudge(self.config.bias_judge)
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        text = observation['text']

        bias_score = self.bias_judge(text)

        self.storage.add(observation)
        self.text_retrieval.add(text)
        self.bias_retrieval.add(bias_score)

# ----- Customize Memory Method -----

class MyMemory(ExplicitMemory):
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = LinearStorage(self.config.args.storage)
        self.insight = {'global_insight': '[None]'}

        self.recall_op = MyMemoryRecall(
            self.config.args.recall,
            storage = self.storage,
            insight = self.insight
        )
        self.store_op = MyMemoryStore(
            self.config.args.store,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            bias_retrieval = self.recall_op.bias_retrieval
        )
        self.optimize_op = RFOptimize(self.config.args.optimize, insight = self.insight)

        self.auto_display = ScreenDisplay(self.config.args.display, register_dict = {
            'Memory Storage': self.storage,
            'Insight': self.insight
        })

    def reset(self):
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])
        self.insight = {'global_insight': '[None]'}

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
    
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        self.optimize_op(**kwargs)

def sample_MyMemory():
    memory_config = MemoryConfig(MyMemoryConfig)
    memory = MyMemory(memory_config)
    trial1 = """Alice: I recently started a fascinating historical fiction book, and I can't put it down!
Assistant: What historical period does it cover?
Alice: It's set during the Renaissance, a time of incredible cultural and intellectual growth.
Assistant: The Renaissance sounds like such a vibrant era! How does the author weave historical facts into the story?"""
    memory.optimize(new_trial = trial1)

    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies?'))


if __name__ == '__main__':
    sample_MyMemory()