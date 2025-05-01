import json

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
    step_info_list = raw_traj['step_info_list']
    question, answer = raw_traj['task_info']['question'], raw_traj['task_info']['answer']
    for sid, step_info in enumerate(step_info_list):
        operation_list = step_info['operation_list']
        for operation_item in operation_list:
            if operation_item['type'] == 'store':
                data_list.append({
                    'raw_observation': operation_item['raw_observation'],
                    'extracted_memory': operation_item['extracted_memory'],
                    'question': question,
                    'answer': answer
                })
    return data_list, raw_traj['reward']
    

def formulate_store_trainset():
    positive_data_list = []
    negative_data_list = []
    path_list = ['Qwen2.5-RUSMemory-%d.jsonl' % index for index in range(4)]
    positive_output_path = 'store_positive.json'
    negative_output_path = 'store_negative.json'
    for path in path_list:
        raw_data_list = read_jsonl(path)
        for tid, raw_traj in enumerate(raw_data_list):
            new_data, sign = extract_from_trajectory(tid, raw_traj)
            if sign == 1:
                positive_data_list.append(new_data)
            elif sign == 0:
                negative_data_list.append(new_data)
            else:
                raise
        
    with open(positive_output_path,'w', encoding='utf-8') as f:
        json.dump(positive_data_list,f, indent=4,ensure_ascii=False)
    
    with open(negative_output_path,'w', encoding='utf-8') as f:
        json.dump(negative_data_list,f, indent=4,ensure_ascii=False)
    
    print(len(positive_data_list), len(negative_data_list))
    # 1326 2658

if __name__ == '__main__':
    formulate_store_trainset()