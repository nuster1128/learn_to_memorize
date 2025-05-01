from abc import ABC, abstractmethod

class BaseUtilization(ABC):
    """
    Deal with several different parts of memory contents, formulating these information into a unified output.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ConcateUtilization(BaseUtilization):
    """
    Concate memory pieces into a string for application.
    """
    def __init__(self, config):
        super().__init__(config)

    def concate_list(self, l):
        if self.config.list_config.index:
            l = ['[%d] %s' % (index, m) for index, m in enumerate(l)]
        if len(l) == 0:
            return 'None'
        return self.config.list_config.sep.join(l)

    def __call__(self, input_memory):
        if isinstance(input_memory, list):
            main_body = self.concate_list(input_memory)
        elif isinstance(input_memory, dict):
            main_body_list = []
            for k,v in input_memory.items():
                if isinstance(v, list):
                    v = self.concate_list(v)
                
                main_body_list.append(self.config.dict_config.key_value_sep.join([self.config.dict_config.key_format % k, v]))
            main_body = self.config.dict_config.item_sep.join(main_body_list)
        elif isinstance(input_memory, str):
            main_body = input_memory
        
        return '\n'.join([self.config.prefix, main_body, self.config.suffix])