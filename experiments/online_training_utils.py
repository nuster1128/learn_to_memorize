import numpy as np
from sentence_transformers import SentenceTransformer
import torch, json, os
from torch.utils.data import TensorDataset, DataLoader
from RUSMemory import ScoreModel, MoEGate
import torch.optim as optim
from openai import OpenAI
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOConfig, DPOTrainer

DECAY = 0.2
MAX_MEMORY_PIECE = 8192

OPENAI_MODEL = 'gpt-4o'
OPENAI_APIKEY = '[API_KEY]'
OPENAI_APIBASE = '[API_BASE]'

def store_data_prepare(batch_data):
    def store_extract_from_trajectory(raw_traj):
        data_list = []
        step_info_list = raw_traj['step_info_list']
        question, answer = raw_traj['task_info']['question'], raw_traj['task_info']['answer']
        for step_info in step_info_list:
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
    
    positive_data_list, negative_data_list = [], []
    for raw_traj in batch_data:
        new_data, sign = store_extract_from_trajectory(raw_traj)
        if sign == 1:
            positive_data_list.append(new_data)
        else:
            negative_data_list.append(new_data)
    
    return positive_data_list, negative_data_list

positive_prompt = """You are an expert who is good at reflection.
I am extracting key information from the raw observation to assist in answering the following question.
Here is one of my successful cases, please help me reflect and summarize the successful experience of the extraction process.

Raw Observation:
{raw_observation}

Extracted Key Information:
{extracted_memory}

Question: {question}
Answer: {answer}

Requirements:
1. The summarized successful experience should be general, rather than containing specific details of this question.
2. The summarized successful experience should be useful and helpful for future information extraction.
3. The summarized successful experience should be concise, usually consisting of one or two sentences.
4. Please only output a summary of successful experience in one line, without providing a thought process or explanation."""

negative_prompt = """You are an expert who is good at reflection.
I am extracting key information from the raw observation to assist in answering the following question.
Here is one of my failed cases, please help me reflect and summarize the failure experience of the extraction process.

Raw Observation:
{raw_observation}

Extracted Key Information:
{extracted_memory}

Question: {question}
Answer: {answer}

Requirements:
1. The summarized failure experience should be general, rather than containing specific details of this question.
2. The summarized failure experience should be useful and helpful for future information extraction.
3. The summarized failure experience should be concise, usually consisting of one or two sentences.
4. Please only output a summary of failure experience in one line, without providing a thought process or explanation."""

summarize_prompt = """You are an expert at summarizing, please help me summarize the following experience into a paragraph.

Experience:
{current_context}
{experience_chunk}

Requirements:
1. The summarized experience should be useful and helpful for future information extraction.
2. Please only output the summary in one line, without providing a thought process or explanation.
3. The summarized experience should be rich, comprehensive, and informative."""

def store_train(positive_data, negative_data, llm, old_hint, chunk_size):
    experience_list = []
    for pd_list in positive_data:
        for pd in pd_list:
            prompt = positive_prompt.format(**pd)
            response = llm.fast_run(prompt)
            experience_list.append(response)

    for pd_list in negative_data:
        for pd in pd_list:
            prompt = positive_prompt.format(**pd)
            response = llm.fast_run(prompt)
            experience_list.append(response)
    
    # chunk_size = 10
    exp_chunk = []
    current_context = old_hint
    for experience in experience_list:
        exp_chunk.append(experience)
        if len(exp_chunk) == chunk_size:
            prompt = summarize_prompt.format(**{
                'experience_chunk': '\n'.join(exp_chunk),
                'current_context': current_context
            })
            current_context = llm.fast_run(prompt)
            exp_chunk = []
        
    return current_context

def retrieval_data_prepare(batch_data):
    class Encoder():
        def __init__(self, model_path):
            self.model = SentenceTransformer(model_path).cuda()
        
        def __call__(self, text_list):
            embeddings = self.model.encode([text_list], normalize_embeddings=True)
            return torch.from_numpy(embeddings).cuda()
        
    def retrieval_extract_from_trajectory(raw_traj):
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
    
    data_list = []

    for raw_traj in batch_data:
        new_data = retrieval_extract_from_trajectory(raw_traj)
        if new_data:
            data_list += new_data

    encoder = Encoder(model_path='[Path]')
    query_text_tensor, core_text_tensor, inverse_text_tenosr = None, None, None
    core_time_list, inverse_time_list, coef_list = [], [], []

    for i in range(len(data_list)):
        query_embedding = encoder(data_list[i]['query_text'])
        core_embedding = encoder(data_list[i]['core_memory_text'])
        inverse_embedding = encoder(data_list[i]['inverse_memory_text'])

        if i == 0:
            query_text_tensor, core_text_tensor, inverse_text_tenosr = query_embedding, core_embedding, inverse_embedding
        else:
            query_text_tensor = torch.vstack((query_text_tensor, query_embedding))
            core_text_tensor = torch.vstack((core_text_tensor, core_embedding))
            inverse_text_tenosr = torch.vstack((inverse_text_tenosr, inverse_embedding))
        
        core_time_list.append(data_list[i]['core_memory_time'])
        inverse_time_list.append(data_list[i]['inverse_memory_time'])
        coef_list.append(data_list[i]['coef'])

    core_time_tensor, inverse_time_tensor, coef_tensor = torch.tensor(core_time_list), torch.tensor(inverse_time_list), torch.tensor(coef_list)
    retrieval_tensor = (query_text_tensor, core_text_tensor, inverse_text_tenosr, core_time_tensor, inverse_time_tensor, coef_tensor)
    
    return retrieval_tensor

def retrieval_train(retrieval_tensor, score_model):
    def calculate_emotion_scores(query, memory, score_model):
        query_emo = score_model.emotion_scorer(query)
        query_emo_norm = torch.norm(query_emo, p=2, dim=1)
        memory_emo = score_model.emotion_scorer(memory)
        memory_emo_norm = torch.norm(memory_emo, p=2, dim=1)
        emotion_scores = torch.sum(torch.mul(memory_emo, query_emo), dim=1) / query_emo_norm / memory_emo_norm
        return emotion_scores

    def score_forward(query, memory, memory_delta_time, score_model):
        moe_gate_score = score_model.moe_gate(query, memory)

        semantic_score = torch.sum(torch.mul(memory, query), dim=1)

        recency_scores = torch.ones(memory_delta_time.size()).unsqueeze(dim=0).cuda()
        for i in range(score_model.time_rank-1):
            recency_scores = torch.cat((recency_scores, (recency_scores[-1] * memory_delta_time).unsqueeze(dim=0)), dim=0)
        recency_scores = - 1 * recency_scores.t()

        emotion_score = calculate_emotion_scores(query, memory, score_model)
        importance_score = score_model.calculate_importance_scores(query, memory)
        scores = torch.cat((semantic_score.unsqueeze(dim=1), recency_scores, emotion_score.unsqueeze(dim=1), importance_score.unsqueeze(dim=1)),dim=1)

        combined_score = torch.mul(moe_gate_score, scores).sum(dim=1)

        return combined_score
    
    train_config = {
        'batch_size': 256,
        'total_epoch': 30,
        'lr': 0.05
    }
    query_text_tensor, core_text_tensor, inverse_text_tenosr, core_time_tensor, inverse_time_tensor, coef_tensor = retrieval_tensor

    train_num = int(query_text_tensor.shape[0] * 0.9)
    # test_num = query_text_tensor.shape[0] - train_num

    train_dataset = TensorDataset(
        query_text_tensor[:train_num].cuda(), core_text_tensor[:train_num].cuda(), core_time_tensor[:train_num].cuda(),
        inverse_text_tenosr[:train_num].cuda(), inverse_time_tensor[:train_num].cuda(), coef_tensor[:train_num].cuda()
    )
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)

    test_dataset = TensorDataset(
        query_text_tensor[train_num:].cuda(), core_text_tensor[train_num:].cuda(), core_time_tensor[train_num:].cuda(),
        inverse_text_tenosr[train_num:].cuda(), inverse_time_tensor[train_num:].cuda(), coef_tensor[train_num:].cuda()
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    optimizer = optim.SGD(score_model.parameters(), lr=train_config['lr'])

    for epoch in range(train_config['total_epoch']):
        train_loss = torch.tensor(0.0)
        for batch_id, (batch_query, batch_core_text, batch_core_time, batch_inverse_text, batch_inverse_time, batch_coef) in enumerate(train_dataloader):
            optimizer.zero_grad()
            core_score = score_forward(batch_query, batch_core_text, batch_core_time, score_model)
            inverse_score = score_forward(batch_query, batch_inverse_text, batch_inverse_time, score_model)

            loss = - torch.sigmoid(core_score - inverse_score) + torch.sigmoid(inverse_score - core_score)

            scaled_loss = torch.mul(loss, batch_coef)
            avg_loss = torch.mean(scaled_loss)

            avg_loss.backward()
            optimizer.step()

            train_loss += avg_loss.detach().cpu()
            
        train_loss /= (batch_id + 1)
        if epoch % 10 == 0:
            print('(Train) Epoch %d loss: %f' % (epoch, train_loss))

        valid_loss = torch.tensor(0.0)
        for batch_id, (batch_query, batch_core_text, batch_core_time, batch_inverse_text, batch_inverse_time, batch_coef) in enumerate(test_dataloader):
            core_score = score_forward(batch_query, batch_core_text, batch_core_time, score_model)
            inverse_score = score_forward(batch_query, batch_inverse_text, batch_inverse_time, score_model)
            loss = - torch.sigmoid(core_score - inverse_score) + torch.sigmoid(inverse_score - core_score)
            scaled_loss = torch.mul(loss, batch_coef)
            avg_loss = torch.mean(scaled_loss)
            
            valid_loss += avg_loss.detach().cpu()
        valid_loss /= (batch_id + 1)
        if epoch % 10 == 0:
            print('------ (Valid) Epoch %d loss: %f' % (epoch, valid_loss))

merge_prompt = """Please merge the above new information into the existing information, in order to make the final information more useful to response to the query.
[Query]
{observation}

[Existing Information]
{memory_context}

[New Information]
{new_memory}

Requirements:
1. You should remove the duplicated information to make it concise, but do not lose any useful information.
2. You should just output the final information after merge, without any other messages. Do not repeat the [Query], [Existing Information] and [New Information]."""

def generate_utilization_data(data_list, llm):
    output_list = []
    for pid in range(len(data_list)):
        item = data_list[pid]
        prompt = merge_prompt.format(**{
                'observation': item['observation'],
                'memory_context': item['memory_context'],
                'new_memory': item['new_memory']
            })
        res = llm.fast_inference(prompt)
        info = {'index': item['index'], 'output': res}
        output_list.append(info)
    return output_list

class OpenAI_LLM():
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

    def fast_inference(self, query):
        response = self.run([{"role": "user", "content": query}])
        max_retry = 5
        while not response['result']:
            max_retry -= 1
            response = self.run([{"role": "user", "content": query}])
            print('LLM Inference Retry.')
            if max_retry == 0:
                return 'None'
        return response['result']

class Wrappper_LLM():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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

def utilization_data_prepare(batch_data):
    def utilization_extract_from_step(sid, query, current_storage_list, ranked_memory_index, intermediate_list):
        ranked_memory_list = [current_storage_list[mid] for mid in ranked_memory_index]

        if not len(ranked_memory_index) == len(intermediate_list):
            print('----- Fail (%d) -----' % sid)
            print(len(ranked_memory_index), len(intermediate_list))
            return False
        data_list = []
        for index in range(len(ranked_memory_list)-1):
            current_memory = intermediate_list[index]
            new_memory = ranked_memory_list[index+1]
            if len(current_memory.split(' ')) >= MAX_MEMORY_PIECE or len(new_memory.split(' ')) >= MAX_MEMORY_PIECE:
                print('----- Too Long to Fail (%d, %d) -----' % (sid, index))
                print(len(current_memory.split(' ')), len(new_memory.split(' ')))
                return False
            data_list.append({
                'observation': query,
                'memory_context': current_memory,
                'new_memory': new_memory
            })
        return data_list
    
    def utilization_extract_from_trajectory(raw_traj):
        data_list = []
        step_info_list = raw_traj['step_info_list']
        for sid, step_info in enumerate(step_info_list):
            if sid != len(step_info_list) - 1:
                continue
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
            
            new_list = utilization_extract_from_step(sid , query_text, current_storage_list, ranked_memory_index, intermediate_list)

            if new_list:
                data_list += new_list

        return data_list

    data_list = []

    for raw_traj in batch_data:
        data_list += utilization_extract_from_trajectory(raw_traj)
    for index, item in enumerate(data_list):
        data_list[index]['index'] = index
    
    expert_llm = OpenAI_LLM({
        'name': OPENAI_MODEL,
        'api_key': OPENAI_APIKEY,
        'api_base': OPENAI_APIBASE
    })
    expert_data = generate_utilization_data(data_list, expert_llm)

    return data_list, expert_data

def map_tokenizer_sft_batch(batch_sample, tokenizer):
    prompt_message_list = [[
            {"role": "user", "content": batch_sample['prompt'][idx]},
        ] for idx in range(len(batch_sample['prompt']))]
    
    prompt_text_list = tokenizer.apply_chat_template(
        prompt_message_list,
        tokenize=False,
        add_generation_prompt=True,
    )

    response_text_list = [
        batch_sample['response'][idx] + tokenizer.eos_token
    for idx in range(len(batch_sample['prompt']))]

    tokenized_prompts = tokenizer(prompt_text_list, return_tensors="pt", max_length=4096, padding='max_length', truncation=True, add_special_tokens=False)
    tokenized_responses = tokenizer(response_text_list, return_tensors="pt", max_length=512, padding='max_length', truncation=True, add_special_tokens=False)

    input_ids = torch.concat([tokenized_prompts['input_ids'], tokenized_responses['input_ids']], dim=1)
    attention_mask = torch.concat([tokenized_prompts['attention_mask'], tokenized_responses['attention_mask']], dim=1)
    labels = torch.tensor([
        [-100] * len(tokenized_prompts['input_ids'][idx]) + tokenized_responses['input_ids'][idx].tolist()
    for idx in range(len(batch_sample['prompt']))], dtype=int)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def prepare_stf_dataset(tokenizer, input_data, raw_expert_data):
    expert_data = []
    for i in range(len(input_data)):
        input_str = merge_prompt.format(**input_data[i])
        expert_data.append({
            'prompt': input_str,
            'response': raw_expert_data[i]['output']
        })
    
    expert_data = Dataset.from_list(expert_data).train_test_split(test_size=0.1)
    expert_data = expert_data.map(map_tokenizer_sft_batch, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    return expert_data

def sft_train(model, tokenizer, dataset, output_path, sft_gb, sft_lr):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)

    train_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft_gb,
        learning_rate=sft_lr,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    trainer.train()

def prepare_dpo_dataset(input_data, expert_data, reference_data):
    dpo_data = []
    for index in range(len(input_data)):
        dpo_data.append({
            "prompt": [{"role": "user", "content": merge_prompt.format(**input_data[index])}],
            "chosen": [{"role": "assistant", "content": expert_data[index]['output']}],
            "rejected": [{"role": "assistant", "content": reference_data[index]['output']}]
        })

    dpo_data = Dataset.from_list(dpo_data).train_test_split(test_size=0.1)
    return dpo_data

def dpo_train(model, tokenizer, dataset, dpo_save_path, dpo_gb, dpo_lr):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)

    dpo_training_args = DPOConfig(
        output_dir=dpo_save_path,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=10
    )
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_training_args,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )
    dpo_trainer.train()

def utilization_train(model, tokenizer, data_list, expert_data, epoch,
                    sft_cache_path, dpo_cache_path, sft_gb, sft_lr, dpo_gb, dpo_lr):
    def get_checkpoint_path(checkpoint_dir):
        file_list = os.listdir(checkpoint_dir)
        max_num = 0
        for file_name in file_list:
            if file_name.startswith('checkpoint'):
                max_num = max(eval(file_name.split('-')[-1]), max_num)
        return f'{checkpoint_dir}/checkpoint-{max_num}'

    # Prepare SFT Dataset
    dataset = prepare_stf_dataset(tokenizer, data_list, expert_data)
    # print(dataset)

    # STF Training
    sft_checkpoint_dir = f'{sft_cache_path}/epoch_{epoch}'
    os.mkdir(sft_checkpoint_dir)
    sft_train(model, tokenizer, dataset, sft_checkpoint_dir, sft_gb, sft_lr)
    
    sft_checkpoint_path = get_checkpoint_path(sft_checkpoint_dir)

    p_model = PeftModel.from_pretrained(model, model_id=sft_checkpoint_path)
    p_model.model.merge_and_unload()

    # Re-generate Reference Dataset
    reference_llm = Wrappper_LLM(p_model, tokenizer)
    reference_data = generate_utilization_data(data_list, reference_llm)

    # Prepare DPO Dataset
    dataset = prepare_dpo_dataset(data_list, expert_data, reference_data)
    # print(dataset)

    # DPO Training
    dpo_checkpoint_dir = f'{dpo_cache_path}/epoch_{epoch}'
    os.mkdir(dpo_checkpoint_dir)
    dpo_train(p_model, tokenizer, dataset, dpo_checkpoint_dir, dpo_gb, dpo_lr)

    dpo_checkpoint_path = get_checkpoint_path(dpo_checkpoint_dir)

    p_model = PeftModel.from_pretrained(p_model, model_id=dpo_checkpoint_path)
    p_model.model.merge_and_unload()
    print(p_model)
    
    return p_model


def save_online_params(final_hint, final_model, final_moe_gate, save_dir):
    # Store Hint Save
    hint_path = f'{save_dir}/hint.json'
    with open(hint_path,'w', encoding='utf-8') as f:
        json.dump({'hint': final_hint},f, indent=4,ensure_ascii=False)
    
    # Utilization Save
    # model_path = f'{save_dir}/online_trained_model'
    # final_model.save_pretrained(model_path)

    # Retrieval Save
    moe_gate_path = f'{save_dir}/trained_moe_gate.pickle'
    torch.save(final_moe_gate, moe_gate_path)

    