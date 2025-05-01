from abc import ABC, abstractmethod
from memengine.function import *

def __recall_convert_str_to_observation__(method):
    """
    If the input is a string, convert it to the dict form.
    """
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper

class BaseRecall(ABC):
    def __init__(self, config):
        self.config = config
    
    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass

    @ abstractmethod
    def __call__(self, query):
        pass

class FUMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        memory_context = self.utilization([obj['text'] for obj in self.storage.get_all_memory_in_order()])
        return self.truncation(memory_context)

class STMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.time_retrieval])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Get the most recent information.
        ranking_ids = self.time_retrieval(query['text'])
        memory_context = self.utilization([self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids])

        return self.truncation(memory_context)

class LTMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Retrieval process.
        ranking_ids = self.text_retrieval(query['text'])
        memory_context = self.utilization([self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids])

        return self.truncation(memory_context)

class GAMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)
        self.importance_retrieval = eval(self.config.importance_retrieval.method)(self.config.importance_retrieval)
        
        self.imporatance_judge = eval(self.config.importance_judge.method)(self.config.importance_judge)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.time_retrieval, self.importance_retrieval, self.imporatance_judge])

    def __retention__(self, target_ids, timestamp):
        """Update the memory of `target_ids` with their recency to `timestamp`.

        Args:
            target_ids (list): the target ids whose memory recency should be updated.
            timestamp (int/float): current timestamp.
        """
        for index in target_ids:
            self.time_retrieval.update(index, timestamp)

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        text = query['text']
        if 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']
        
        # Calculate weighted retrieval scores.
        text_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
        recency_scores, _ = self.time_retrieval(timestamp, topk=False, with_score = True, sort = False)
        importance_scores, _ = self.importance_retrieval(None, topk=False, with_score = True, sort = False)

        score_metrix = torch.stack([text_scores, recency_scores, importance_scores], dim=1)
        comprehensive_scores = torch.matmul(score_metrix, torch.ones(3).to(self.text_retrieval.encoder.device))

        scores, ranking_ids = torch.sort(comprehensive_scores, descending=True)

        if hasattr(self.config, 'topk'):
            scores, ranking_ids = scores[:self.config.topk], ranking_ids[:self.config.topk]
        
        # Update the recency of retrieved memory.
        self.__retention__(ranking_ids, timestamp)

        memory_context = self.utilization([self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids])

        return self.truncation(memory_context)


class MBMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.summary = {'global': 'None'}

        if hasattr(self.config, 'forget'):
            self.forget = MBForget(self.config.forget)

    def __retention__(self, mid, timestamp):
        """Update the memory of `mid` with its recency to `timestamp`.

        Args:
            mid (int): the memory id whose memory recency should be updated.
            timestamp (int/float): current timestamp.
        """
        self.storage.update_memory_attribute_by_mid(mid, 'time', timestamp)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval])
        self.summary = {'global': 'None'}

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        text = query['text']
        if 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']

        # Add the global summary into the memory context.
        memory_context_list = [self.summary['global']]
        
        # Retrieval process.
        ranking_ids = self.text_retrieval(text, topk=False)

        # Ignore certain memory by forggeting according to their recencies and strengths.
        for mid in ranking_ids:
            text = self.storage.get_memory_text_by_mid(mid)
            recency = self.storage.get_memory_attribute_by_mid(mid, 'time')
            strength = self.storage.get_memory_attribute_by_mid(mid, 'strength')
            
            if hasattr(self.config, 'forget'):
                if self.forget.sample_forget(timestamp, recency, strength):
                    continue
            memory_context_list.append(text)

        memory_context = self.utilization(memory_context_list)
        return self.truncation(memory_context)


class SCMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)

        self.summarizer = eval(self.config.summarizer.method)(self.config.summarizer)
        self.activation_judge = eval(self.config.activation_judge.method)(self.config.activation_judge)
        self.summary_judge = eval(self.config.summary_judge.method)(self.config.summary_judge)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.time_retrieval, self.summarizer, self.activation_judge, self.summary_judge])
    
    def __get_flash_memory__(self, timestamp):
        """Get the most recent information.

        Args:
            timestamp (int/float): current timestamp.

        Returns:
            list: the list of recent memory items.
        """
        flash_ids = self.time_retrieval(timestamp, topk=self.config.flash_capacity)
        flash_memory_list = [self.storage.get_memory_text_by_mid(mid) for mid in flash_ids]
        return flash_memory_list

    def __check_require_activation__(self, text, flash_memory_list):
        """Check whether needs to retrieve more information besides flash memory.

        Args:
            text (str): query or observation.
            flash_memory_list (list): the list of flash memory.

        Returns:
            bool: whether needs to retrieve more information besides flash memory.
        """
        return self.activation_judge({
            'query': text,
            'flash_memory': self.utilization.concate_list(flash_memory_list)
        }, post_process = 'bool')
    
    def __check_require_summary__(self, text, activation_summary_list, flash_memory_list):
        """Check whether can just use summary of the retrieved information rather than full texts.

        Args:
            text (string): query or observation.
            activation_summary_list (list): list of retrieved information after summarization.
            flash_memory_list (list): the list of flash memory.

        Returns:
            bool: whether can just use summary of the retrieved information rather than full texts.
        """
        return self.summary_judge({
            'query': text,
            'activation_summary': self.utilization.concate_list(activation_summary_list),
            'flash_memory': self.utilization.concate_list(flash_memory_list)
        }, post_process = 'bool')

    def __retention__(self, target_ids, timestamp):
        """Update the memory of `target_ids` with their recency to `timestamp`.

        Args:
            target_ids (list): the target ids whose memory recency should be updated.
            timestamp (int/float): current timestamp.
        """
        for index in target_ids:
            self.time_retrieval.update(index, timestamp)

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        text = query['text']

        if 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']

        # Get flash memory.
        flash_memory_list = self.__get_flash_memory__(timestamp)

        # Check whether needs to retrieve inforation.
        if not self.__check_require_activation__(text, flash_memory_list):
            memory_context = self.utilization(flash_memory_list)
            return self.truncation(memory_context)
        else:
            # Retrival process.
            text_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
            recency_scores, _ = self.time_retrieval(timestamp, topk=False, with_score = True, sort = False)

            score_metrix = torch.stack([text_scores, recency_scores], dim=1)
            comprehensive_scores = torch.matmul(score_metrix, torch.ones(2).to(self.text_retrieval.device))

            scores, ranking_ids = torch.sort(comprehensive_scores, descending=True)

            if hasattr(self.config, 'activation_topk'):
                activation_ids = ranking_ids[:self.config.activation_topk]
            
            # Update the recency of the retrieved memory.
            self.__retention__(activation_ids, timestamp)

            # Summarize the retrieved memory for each.
            activation_summary_list = [self.summarizer({'content': self.storage.get_memory_text_by_mid(mid)})
                    for mid in activation_ids]

            # Check whether can just utilize summary of each memory rather than full texts.
            if self.__check_require_summary__(text, activation_summary_list, flash_memory_list):
                memory_context = self.utilization({
                    'History Summary': activation_summary_list,
                    'Flash Memory': flash_memory_list
                })
            else:
                memory_context = self.utilization({
                    'History Memory': [self.storage.get_memory_text_by_mid(mid) for mid in activation_ids],
                    'Flash Memory': flash_memory_list
                })
            
            return self.truncation(memory_context)

class MGMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

        self.main_context = kwargs['main_context']
        self.recall_storage = kwargs['recall_storage']
        self.archival_storage = kwargs['archival_storage']

        self.recall_retrieval = eval(self.config.recall_retrieval.method)(self.config.recall_retrieval)
        self.archival_retrieval = eval(self.config.archival_retrieval.method)(self.config.archival_retrieval)
        self.trigger = LLMTrigger(self.config.trigger)
        self.warning_number = int(self.config.warning_threshold * self.config.truncation.number)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.recall_retrieval, self.archival_retrieval, self.trigger])

    def __get_current_memory_prompt(self):
        """
        To provide the current state of memory model for constructing prompts.
        """
        piece_num, piece_mode = self.truncation.get_piece_number(self.get_current_memory_context()), self.config.truncation.mode
        working_memory = '\n'.join(['[%d] %s' % (wid, wtext['text']) for wid, wtext in enumerate(self.main_context['working_context'].get_all_memory_in_order())])
        FIFO_memory = '\n'.join(['[%d] %s' % (fid, ftext['text']) for fid, ftext in enumerate(self.main_context['FIFO_queue'].get_all_memory_in_order())])
        prompt = """Total Capacity: %s %ss
Working Memory(capacity %s words):
%s
Recursive Memory Summary:
%s
FIFO Memory:
%s
""" % (piece_num, piece_mode, self.config.truncation.number, working_memory, self.main_context['recursive_summary']['global'], FIFO_memory)

        return prompt

    def get_current_memory_context(self):
        memory_context = self.utilization({
            'Working Memory': [element['text'] for element in self.main_context['working_context'].get_all_memory_in_order()],
            'Recursive Memory Summary': self.main_context['recursive_summary']['global'],
            'FIFO Memory': [element['text'] for element in self.main_context['FIFO_queue'].get_all_memory_in_order()]
        })
        return memory_context

    def __trigger_function_call__(self, text):
        """Trigger extensible functions.

        Args:
            text (str): query or observation.
        """
        warning_flag = self.truncation.get_piece_number(text) > self.warning_number

        info_dict = {}
        
        info_dict['text'] = text
        info_dict['memory_prompt'] = self.__get_current_memory_prompt()

        # Insert warning sentences.
        if warning_flag:
            info_dict['warning_content'] = self.config.warning_content + '\n'
        else:
            info_dict['warning_content'] = ''

        # Allow trigger none function.
        if hasattr(self.config.trigger, 'no_execute'):
            info_dict['no_execute_prompt'] = self.config.trigger.no_execute + '\n'
        else:
            info_dict['no_execute_prompt'] = ''

        # Add few-shot examples.
        info_dict['few_shot'] = self.config.trigger.few_shot

        # Get trigger results, parse and execute them.
        exe_list = self.trigger(info_dict)
        for (func_name, func_args) in exe_list:
            try:
                execute_command = 'self.__%s__' % func_name
                eval(execute_command)(*func_args)
                print('Successfully execute Function [%s(%s)]' % (func_name, func_args))
            except Exception as e:
                print('Fail to execute Function [%s(%s)]: %s'% (func_name, func_args, e))

    def __memory_retrieval__(self, query):
        """Retrieve query-related information from (external) archival storage, and add the result into working memory.

        Args:
            query (str): a string to retrieve relevant information (e.g., \'Alice\'s name).

        Returns:
            list: list of indexes to be added into working memory.
        """
        if self.archival.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.archival_retrieval(text)
        for mid in ranking_ids:
            self.main_context['working_context'].add(self.archival_storage.get_memory_element_by_mid(mid))
    
    def __memory_recall__(self, query):
        """retrieve query-related information from (external) recall storage, and add the result into FIFO memory.

        Args:
            query (str): a string to retrieve relevant information (e.g., \'Alice\'s name).

        Returns:
            list: list of indexes to be added into FIFO memory.
        """
        if self.recall_storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.recall_retrieval(text)
        for mid in ranking_ids:
            self.main_context['FIFO_queue'].add(self.recall_storage.get_memory_element_by_mid(mid))

    def __memory_archive__(self, memory_list):
        """Archive some memory from FIFO memory into (external) archival storage.

        Args:
            memory_list (list): the index list of FIFO memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.recall_storage.add(element)
            self.recall_retrieval.add(element['text'])

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_transfer__(self, memory_list):
        """Transfer some memory from FIFO memory into working memory.

        Args:
            memory_list (list): the index list of FIFO memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.main_context['working_context'].add(element)

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_save__(self, memory_list):
        """Archive some memory from working memory into (external) archival storage.

        Args:
            memory_list (list): the index list of working memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['working_context'].get_memory_element_by_mid(mid)
            self.archival_storage.add(element)
            self.archival_retrieval.add(element['text'])

        self.main_context['working_context'].delete_by_mid_list(memory_list)
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        text = query['text']

        # Trigger functions to update the current state of memory model.
        self.__trigger_function_call__(text)

        return self.get_current_memory_context()
    
class RFMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.insight = kwargs['insight']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            if self.insight['global_insight']:
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight']
                })
            else:
                return self.config.empty_memory
        else:
            if self.insight['global_insight']:
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight'],
                    'Memory': [obj['text'] for obj in self.storage.get_all_memory_in_order()]
                })
            else:
                memory_context = self.utilization([obj['text'] for obj in self.storage.get_all_memory_in_order()])
        
        return self.truncation(memory_context)
