import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DECAY = 0.2

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                json_obj = json.loads(stripped_line)
                data.append(json_obj)
    return data


def extract_from_trajectory(tid, raw_traj):
    data_list = []
    coef_list = []
    if raw_traj['reward'] == 0:
        return False
    
    last_step_info = raw_traj['step_info_list'][-1]
    operation_list = last_step_info['operation_list']
    for operation_item in operation_list:
        if operation_item['type'] == 'retrieval':
            current_time, normal_time = operation_item['current_time'], operation_item['normal_time']
            query_text = operation_item['query_text']
            memory_text = operation_item['memory_text']
            memory_time = operation_item['memory_time']
            ranked_memory_index = operation_item['ranked_memory_index']
            memory_length = len(ranked_memory_index)
            for core_index in range(memory_length):
                inverse_index = memory_length - 1 - core_index
                coef = core_index - inverse_index
                # print('----- Core %d -----' % core_index)
                # print('Coef:',coef)
                # print('Query:', query_text)
                # print('Core Memory Text:', memory_text[ranked_memory_index[core_index]])
                # print('Core Memory Time:', (current_time - memory_time[ranked_memory_index[core_index]]) / normal_time)
                # print('Inverse Memory Text:', memory_text[ranked_memory_index[inverse_index]])
                # print('Inverse Memory Time:', (current_time - memory_time[ranked_memory_index[inverse_index]]) / normal_time)
                if coef != 0:
                    coef_list.append(coef)
                    data_list.append({
                        'query_text': query_text,
                        'core_memory_text': memory_text[ranked_memory_index[core_index]],
                        'core_memory_time': (current_time - memory_time[ranked_memory_index[core_index]]) / normal_time,
                        'inverse_memory_text': memory_text[ranked_memory_index[inverse_index]],
                        'inverse_memory_time': (current_time - memory_time[ranked_memory_index[inverse_index]]) / normal_time
                    })
    coef_list = np.array(coef_list)
    sign_list = np.sign(coef_list) * -1
    value_list = (memory_length - 1 - np.abs(coef_list)) / 2.0
    weight_list = np.power(DECAY, value_list)
    total_sum = np.sum(weight_list)
    coef_list =  weight_list* sign_list / total_sum
    for i in range(len(data_list)):
        data_list[i]['coef'] = coef_list[i]
    
    return data_list

def formulate_retrieval_trainset():
    data_list = []
    path_list = ['Qwen2.5-RUSMemory-%d.jsonl' % index for index in range(4)]
    output_path = 'retrieval_contrast.json'
    for path in path_list:
        raw_data_list = read_jsonl(path)
        for tid, raw_traj in enumerate(raw_data_list):
            new_data = extract_from_trajectory(tid, raw_traj)
            if new_data:
                data_list += new_data
        
    with open(output_path,'w', encoding='utf-8') as f:
        json.dump(data_list,f, indent=4,ensure_ascii=False)
    
    print(len(data_list))
    # 2402

def load_json(path):
    with open(path,'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class Encoder():
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path).cuda()
    
    def __call__(self, text_list):
        embeddings = self.model.encode([text_list], normalize_embeddings=True)
        return torch.from_numpy(embeddings).cuda()

encoder = Encoder(model_path='[Path]')

def generate_retrieval_tensor():
    input_path = 'retrieval_contrast.json'
    output_path = 'retrieval_tensor.pickle'
    input_data_list = load_json(input_path)

    query_text_tensor, core_text_tensor, inverse_text_tenosr = None, None, None
    core_time_list, inverse_time_list, coef_list = [], [], []

    for i in range(len(input_data_list)):
        query_embedding = encoder(input_data_list[i]['query_text'])
        core_embedding = encoder(input_data_list[i]['core_memory_text'])
        inverse_embedding = encoder(input_data_list[i]['inverse_memory_text'])

        if i == 0:
            query_text_tensor, core_text_tensor, inverse_text_tenosr = query_embedding, core_embedding, inverse_embedding
        else:
            query_text_tensor = torch.vstack((query_text_tensor, query_embedding))
            core_text_tensor = torch.vstack((core_text_tensor, core_embedding))
            inverse_text_tenosr = torch.vstack((inverse_text_tenosr, inverse_embedding))
        
        core_time_list.append(input_data_list[i]['core_memory_time'])
        inverse_time_list.append(input_data_list[i]['inverse_memory_time'])
        coef_list.append(input_data_list[i]['coef'])

        print(i, 'Finish!')

    core_time_tensor, inverse_time_tensor, coef_tensor = torch.tensor(core_time_list), torch.tensor(inverse_time_list), torch.tensor(coef_list)
    print(query_text_tensor.shape, core_text_tensor.shape, inverse_text_tenosr.shape)
    print(core_time_tensor.shape, inverse_time_tensor.shape, coef_tensor.shape)
    # torch.Size([2402, 768]) torch.Size([2402, 768]) torch.Size([2402, 768])
    # torch.Size([2402]) torch.Size([2402]) torch.Size([2402])
    
    # print(coef_tensor)

    retrieval_tensor = (query_text_tensor, core_text_tensor, inverse_text_tenosr, core_time_tensor, inverse_time_tensor, coef_tensor)
    torch.save(retrieval_tensor,output_path)


if __name__ == '__main__':
    formulate_retrieval_trainset()
    generate_retrieval_tensor()