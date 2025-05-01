from abc import ABC, abstractmethod
from memengine.function import *
import torch
import numpy as np

def __store_convert_str_to_observation__(method):
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper

class BaseStore(ABC):
    def __init__(self, config):
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass
    
    @ abstractmethod
    def __call__(self, observation):
        pass

class FUMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        self.storage.add(observation)

class STMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.time_retrieval = kwargs['time_retrieval']

    def reset(self):
        pass
    
    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.storage.counter
        else:
            timestamp = observation['time']
        self.storage.add(observation)
        self.time_retrieval.add(timestamp)

class LTMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        text = observation['text']
        self.storage.add(observation)
        self.text_retrieval.add(text)

class GAMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.time_retrieval = kwargs['time_retrieval']
        self.importance_retrieval = kwargs['importance_retrieval']
        self.imporatance_judge = kwargs['imporatance_judge']
        self.reflector = kwargs['reflector']

    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.storage.counter
        else:
            timestamp = observation['time']
        text = observation['text']

        # Judge importance score for the observation.
        importance_score = self.imporatance_judge({'message': text})

        if 'source' not in observation:
            observation['source'] = False

        # Take a reflection update immediately after storing the observation.
        self.reflector.add_reflection(importance_score, observation['source'])

        self.storage.add(observation)
        self.time_retrieval.add(timestamp)
        self.text_retrieval.add(text)
        self.importance_retrieval.add(importance_score)
        
class MBMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.summary = kwargs['summary']
        
        self.summarizer = eval(self.config.summarizer.method)(self.config.summarizer)
    
    def reset(self):
        self.__reset_objects__([self.summarizer])

    def __summarize_history__(self, time):
        """Summarize the information inside the certain time block.

        Args:
            time (int/float): time identifier.
        """
        mids = self.storage.get_mids_by_attribute('time', time)
        histories = '\n'.join([self.storage.get_memory_text_by_mid(mid) for mid in mids])
        current_summary = self.summarizer({'content': histories})
        self.summary[time] = current_summary

        self.storage.add({
            'text': current_summary,
            'time': time,
            'source': 'summary',
            'strength': 1.0
        })
        self.text_retrieval.add(current_summary)

    def __summarize_summary__(self):
        """
        Update the global summary.
        """
        summaries = [summary_text for summary_time, summary_text in self.summary.items() if summary_time != 'global']
        self.summary['global'] = self.summarizer({'content': summaries})

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if not self.storage.is_empty():
            last_timestamp = self.storage.get_memory_attribute_by_mid(-1, 'time')
            last_source = self.storage.get_memory_attribute_by_mid(-1, 'source')
        else:
            last_timestamp = None
            last_source = None

        if 'time' not in observation:
            timestamp = self.storage.counter
            observation['time'] = timestamp
        else:
            timestamp = observation['time']
        text = observation['text']

        if 'source' not in observation:
            observation['source'] = 'history'
        observation['strength'] = 1.0
        
        self.storage.add(observation)
        self.text_retrieval.add(text)

        # Check whether needs summarize current information.
        if timestamp != last_timestamp and last_source == 'history':
            self.__summarize_history__(last_timestamp)
            self.__summarize_summary__()


class SCMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.time_retrieval = kwargs['time_retrieval']
        self.summarizer = kwargs['summarizer']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.storage.counter
            observation['time'] = timestamp
        else:
            timestamp = observation['time']
        
        text = observation['text']
        
        if 'summary' not in observation:
            summary = self.summarizer({'content': text})
            observation['summary'] = summary
        else:
            summary = observation['summary']

        self.storage.add(observation)
        self.time_retrieval.add(timestamp)
        self.text_retrieval.add(text)

class MGMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.main_context = kwargs['main_context']
        self.recall_storage = kwargs['recall_storage']
        self.recall_retrieval = kwargs['recall_retrieval']
        self.truncation = kwargs['truncation']

        self.summarizer = eval(self.config.summarizer.method)(self.config.summarizer)
        self.flush_checker = eval(self.config.flush_checker.method)(self.config.flush_checker)
    
    def reset(self):
        self.__reset_objects__([self.summarizer, self.flush_checker])

    def __flush_queue__(self):
        """
        Flush the FIFO queue.
        """
        FIFO_queue = self.main_context['FIFO_queue'].get_all_memory_in_order()

        flush_context = ''
        for mid, element in enumerate(FIFO_queue):
            flush_context += '\n%s' % element['text']
            self.recall_storage.add(element)
            self.recall_retrieval.add(element['text'])
            if self.flush_checker.check_truncation_needed(flush_context):
                break
        
        self.main_context['recursive_summary']['global'] = self.summarizer({
            'recursive_summary': self.main_context['recursive_summary']['global'],
            'flush_context': flush_context
        })

        self.main_context['FIFO_queue'].clear_memory(start=0, end=mid+1)

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.main_context['FIFO_queue'].counter
            observation['time'] = timestamp
        else:
            timestamp = observation['time']
        
        text = observation['text']
        
        self.main_context['FIFO_queue'].add(observation)

        # Check the state of FIFO queue to determine whether needs to flush it.
        if self.flush_checker.check_truncation_needed(text):
            self.__flush_queue__()

class MTMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.max_depth = 0
        self.summarizer = eval(self.config.summarizer.method)(self.config.summarizer)
    
    def reset(self):
        self.__reset_objects__([self.summarizer])

    def __get_traverse_threshold__(self, d):
        return self.config.traverse_base_threshold * np.exp(self.config.traverse_rate * d / self.max_depth)

    def __recursive_insert_node__(self, new_node_id, text_embedding):
        current_node_id= 0
        while True:
            memory_element = self.storage.get_memory_element_by_node_id(current_node_id)
            child = memory_element['child']
            # Check whether it is a leaf node.
            if len(child) == 0:
                self.storage.update_memory_attribute_by_node_id(new_node_id, 'degree', memory_element['degree'] + 1)
                self.storage.update_memory_attribute_by_node_id(new_node_id, 'parent', current_node_id)
                self.storage.update_memory_attribute_by_node_id(current_node_id, 'child', memory_element['child'] + [new_node_id])
                self.max_depth = max(self.max_depth, memory_element['degree'] + 1)
                return
            child_embeddings = self.text_retrieval.get_tensor_by_ids(child)
            scores = torch.matmul(text_embedding,child_embeddings.T).squeeze(0)
            max_value, max_index = torch.max(scores, dim=0)

            # Compare the maximum traverse value with the threshold.
            if max_value < self.__get_traverse_threshold__(memory_element['degree']):
                self.storage.update_memory_attribute_by_node_id(new_node_id, 'degree', memory_element['degree'] + 1)
                self.storage.update_memory_attribute_by_node_id(new_node_id, 'parent', current_node_id)
                self.storage.update_memory_attribute_by_node_id(current_node_id, 'child', memory_element['child'] + [new_node_id])
                self.max_depth = max(self.max_depth, memory_element['degree'] + 1)
                return
            else:
                # Add information into the current node before next traversing.
                summary = self.summarizer({
                    'n_children': len(child),
                    'new_content': self.storage.get_memory_text_by_node_id(new_node_id),
                    'current_content': memory_element['text']
                })
                self.storage.update_memory_attribute_by_node_id(current_node_id, 'text', summary)
                mid = self.storage.get_mid_by_node_id(current_node_id)
                self.text_retrieval.update(mid, summary)

                # Continue to traverse.
                current_node_id = child[max_index]

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if self.storage.is_empty():
            self.storage.add_node({
                'text':'[ROOT]',
                'time': self.storage.node_counter,
                'degree': 0,
                'parent': None,
                'child': []
            })

        if 'time' not in observation:
            timestamp = self.storage.node_counter
            observation['time'] = timestamp
        else:
            timestamp = observation['time']
        
        text = observation['text']
        observation['parent'] = None
        observation['child'] = []
        observation['degree'] = 0
        
        node_id = self.storage.add_node(observation)
        text_embedding = self.text_retrieval.add(text)

        # Attach the node recursively.
        self.__recursive_insert_node__(node_id,text_embedding)
