from abc import ABC, abstractmethod
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate
import re

class BaseTrigger(ABC):
    """
    Designed to call functions or tools in extensible manners.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LLMTrigger(BaseTrigger):
    """
    Utilizing LLMs to determine which function should be called with specific arguments. 
    """
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)
        self.func_dict = {}

        for func in config.func_list:
            self.register(func)

    def register(self, func):
        new_func = {
            'name': func['name'],
            'args': func['args'],
            'args_type': func['args_type'],
            'func_description': func['func_description'],
            'args_description': func['args_description']
        }
        self.func_dict[func['name']] = new_func

    def __get_function_prompt__(self):
        function_content = '\n\n'.join(["""def %s(%s):
    Description: %s
    Args: %s""" % (func['name'], ','.join(func['args']), func['func_description'], ';'.join(func['args_description']))
        for func_name, func in self.func_dict.items()])
        return """----- Function Descriptions Start -----
%s
----- Function Descriptions End -----""" % function_content

    def __parse_excuate_function__(self, res):
        if hasattr(self.config, 'no_execuate'):
            if res.strip() == self.config.no_execuate:
                return False
        excuate_list = [q for q in res.splitlines() if q.strip() != '']
        return_list = []
        # print('[Res]:: %s' % res)
        for exe_text in excuate_list:
            pattern = r'(\w+)\((.*?)\)'
            match = re.search(pattern, exe_text)
            if match:
                func_name, func_args = match.group(1), match.group(2)
                parsed_args = []
                # Previous Code
                # func_arg_list = func_args.split(',')
                # New Code
                func_arg_list = ['%s' % item for item in eval('[%s]' % func_args)]
                # print(func_arg_list)
                if func_name in self.func_dict:
                    # print('[func_name]:: %s' % func_name)
                    # print('[defined args]:: %s' % self.func_dict[func_name]['args'])
                    # print('[real args]:: %s' % func_arg_list)
                    assert len(func_arg_list) == len(self.func_dict[func_name]['args'])
                    for ind, tp in enumerate(self.func_dict[func_name]['args_type']):
                        if tp in ['str']:
                            parsed_args.append(eval(tp)(func_arg_list[ind]))
                        else:
                            parsed_args.append(eval(func_arg_list[ind]))
                        
                return_list.append((func_name, parsed_args))
            else:
                print('Execuate Parse Fail.')
        return return_list

    def __call__(self, input_dict):
        input_dict['function_prompt'] = self.__get_function_prompt__()
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        for k,v in input_dict.items():
            print('[%s]:: %s' % (k,v))
        
        # print('[input_variables]:: %s' % self.config.prompt.input_variables)
        # print('[template]:: %s' % self.config.prompt.template)


        prompt = prompt_template.format(**input_dict)
        # print(prompt)
        res = self.llm.fast_run(prompt)

        return self.__parse_excuate_function__(res)

