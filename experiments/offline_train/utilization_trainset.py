from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json, os, requests, torch
from openai import OpenAI
from langchain.prompts import PromptTemplate

OPENAI_MODEL = 'gpt-4o'
OPENAI_APIKEY = '[API_KEY]'
OPENAI_APIBASE = '[API_BASE]'

MAX_MEMORY_PIECE = 8196

class LLM():
    def __init__(self, llm_config):
        self.llm_config = llm_config

        self.client = OpenAI(api_key=self.llm_config['api_key'], base_url=self.llm_config['api_base'])

    def parse_response(self, response):
        return {'result': response.choices[0].message.content}

    def run(self, message_list):
        response = self.client.chat.completions.create(
            model=self.llm_config['name'],
            messages=message_list,
            temperature=0.9
        )
        response = self.parse_response(response)
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

class SFT_LLM():
    def __init__(self, llm_config):

        self.model_path = llm_config['model_path']
        self.lora_path = llm_config['lora_path']

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda", torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = PeftModel.from_pretrained(self.model, model_id=self.lora_path)

    def inference(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=4096,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, max_length=4096, padding='max_length', truncation=True)[0]
        
        return response
    
    def fast_inference(self, query):
        response = self.inference([{"role": "user", "content": query}])
        return response

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                json_obj = json.loads(stripped_line)
                data.append(json_obj)
    return data

def extract_from_step(tid, sid, query, current_storage_list, ranked_memory_index, intermediate_list):
    ranked_memory_list = [current_storage_list[mid] for mid in ranked_memory_index]

    if not len(ranked_memory_index) == len(intermediate_list):
        print('----- Fail (%d, %d) -----' % (tid, sid))
        print(len(ranked_memory_index), len(intermediate_list))
        return False
    data_list = []
    for index in range(len(ranked_memory_list)-1):
        current_memory = intermediate_list[index]
        new_memory = ranked_memory_list[index+1]
        if len(current_memory.split(' ')) >= MAX_MEMORY_PIECE or len(new_memory.split(' ')) >= MAX_MEMORY_PIECE:
            print('----- Too Long to Fail (%d, %d, %d) -----' % (tid, sid, index))
            print(len(current_memory.split(' ')), len(new_memory.split(' ')))
            return False
        # print('----- Index %d -----' % index)
        # print('Current Memory:', current_memory)
        # print('New Memory:', new_memory)
        data_list.append({
            'observation': query,
            'memory_context': current_memory,
            'new_memory': new_memory
        })
    return data_list

def extract_from_trajectory(tid, raw_traj):
    data_list = []
    step_info_list = raw_traj['step_info_list']
    for sid, step_info in enumerate(step_info_list):
        # ----- This block will make it only using the last step of each trajectory. -----
        if sid != len(step_info_list) - 1:
            continue
        # ----- End -----
        operation_list = step_info['operation_list']
        current_storage_list, ranked_memory_index, intermediate_list, query_text = None, None, None, None
        for operation_item in operation_list:
            if operation_item['type'] == 'retrieval':
                current_storage_list = operation_item['memory_text']
                ranked_memory_index = operation_item['ranked_memory_index']
                query_text = operation_item['query_text']
            if operation_item['type'] == 'utilization':
                intermediate_list = operation_item['intermediate_list']
        if len(current_storage_list) == 1:
            continue
        
        new_list = extract_from_step(tid, sid , query_text, current_storage_list, ranked_memory_index, intermediate_list)

        if new_list:
            data_list += new_list

    return data_list

def formulate_utilization_trainset():
    data_list = []
    path_list = ['Qwen2.5-RUSMemory-%d.jsonl' % index for index in range(4)]
    output_path = 'utilization_input.json'
    for path in path_list:
        raw_data_list = read_jsonl(path)
        for tid, raw_traj in enumerate(raw_data_list):
            data_list += extract_from_trajectory(tid, raw_traj)
    for index, item in enumerate(data_list):
        data_list[index]['index'] = index
    
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list,f, indent=4,ensure_ascii=False)
    
    print(len(data_list))
    # 4484

def load_json(path):
    with open(path,'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_expert_data():
    llm = LLM({
        'name': OPENAI_MODEL,
        'api_key': OPENAI_APIKEY,
        'api_base': OPENAI_APIBASE
    })
    input_path = 'utilization_input.json'
    input_data_list = load_json(input_path)
    ouput_path = 'utilization_output_gpt4o.jsonl'
    start_index = 0
    for pid in range(start_index, len(input_data_list)):
        item = input_data_list[pid]
        prompt = PromptTemplate(
                input_variables=['observation', 'memory_context', 'new_memory'],
                template="""Please merge the above new information into the existing information, in order to make the final information more useful to response to the query.
[Query]
{observation}

[Existing Information]
{memory_context}

[New Information]
{new_memory}

Requirements:
1. You should remove the duplicated information to make it concise, but do not lose any useful information.
2. You should just output the final information after merge, without any other messages. Do not repeat the [Query], [Existing Information] and [New Information].""",
            ).format(**{
                'observation': item['observation'],
                'memory_context': item['memory_context'],
                'new_memory': item['new_memory']
            })
        res = llm.fast_run(prompt)
        info = {'index': item['index'], 'output': res}
        with open(ouput_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(info,ensure_ascii=False)
            f.write(json_line + '\n')
        print(item['index'],'Finish!')

def generate_reference_data():
    llm = SFT_LLM(llm_config={
        'model_path': '[Path for basemodel]',
        'lora_path': '[Path for Lora checkpoint]'
    })

    input_path = 'utilization_input.json'
    input_data_list = load_json(input_path)
    ouput_path = 'utilization_output_reference.jsonl'
    start_index = 0
    for pid in range(start_index, len(input_data_list)):
        item = input_data_list[pid]
        prompt = PromptTemplate(
                input_variables=['observation', 'memory_context', 'new_memory'],
                template="""Please merge the above new information into the existing information, in order to make the final information more useful to response to the query.
[Query]
{observation}

[Existing Information]
{memory_context}

[New Information]
{new_memory}

Requirements:
1. You should remove the duplicated information to make it concise, but do not lose any useful information.
2. You should just output the final information after merge, without any other messages. Do not repeat the [Query], [Existing Information] and [New Information].""",
            ).format(**{
                'observation': item['observation'],
                'memory_context': item['memory_context'],
                'new_memory': item['new_memory']
            })
        res = llm.fast_run(prompt)
        info = {'index': item['index'], 'output': res}
        with open(ouput_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(info,ensure_ascii=False)
            f.write(json_line + '\n')
        print(item['index'],'Finish!')


if __name__ == '__main__':
    formulate_utilization_trainset()
    generate_expert_data()
    generate_reference_data()
    