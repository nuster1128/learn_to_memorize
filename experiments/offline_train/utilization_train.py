import json, time
import numpy as np
import torch, os
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, Qwen2ForCausalLM, Qwen2Tokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOConfig, DPOTrainer
from copy import deepcopy

model_path = '[Path]'

def load_json(path):
    with open(path,'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

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


def inference(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, max_length=4096, padding='max_length', truncation=True)[0]
    return response

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

def prepare_stf_dataset(tokenizer):
    input_path = 'utilization_input.json'
    output_path = 'utilization_output_gpt4o.jsonl'

    output_data = load_jsonl(output_path)
    # total_num = len(output_data)
    total_num = 4000
    input_data = load_json(input_path)[:total_num]
    
    expert_data = []
    for i in range(total_num):
        input_str = merge_prompt.format(**input_data[i])
        expert_data.append({
            'prompt': input_str,
            'response': output_data[i]['output']
        })
    
    expert_data = Dataset.from_list(expert_data).train_test_split(test_size=0.1)
    expert_data = expert_data.map(map_tokenizer_sft_batch, fn_kwargs={'tokenizer': tokenizer}, batched=True)
    return expert_data

def sft_train(model, tokenizer, dataset, output_path, config):
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
        gradient_accumulation_steps=config['sft_gradient_steps'],
        learning_rate=config['sft_lr'],
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

def generate_reference_data(model, tokenizer):
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
        res = inference(model, tokenizer, [{"role": "user", "content": prompt}])
        info = {'index': item['index'], 'output': res}
        with open(ouput_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(info,ensure_ascii=False)
            f.write(json_line + '\n')
        print(item['index'],'Finish!')

def prepare_dpo_dataset():
    # Max: 4484
    total_num = 4000
    input_path = 'utilization_input.json'
    expert_path = 'utilization_output_gpt4o.jsonl'
    reference_path = 'utilization_output_reference.jsonl'

    input_data = load_json(input_path)
    expert_data = load_jsonl(expert_path)
    reference_data = load_jsonl(reference_path)

    dpo_data = []
    for index in range(len(input_data[:total_num])):
        dpo_data.append({
            "prompt": [{"role": "user", "content": merge_prompt.format(**input_data[index])}],
            "chosen": [{"role": "assistant", "content": expert_data[index]['output']}],
            "rejected": [{"role": "assistant", "content": reference_data[index]['output']}]
        })
        # print(dpo_data)

    dpo_data = Dataset.from_list(dpo_data).train_test_split(test_size=0.1)

    return dpo_data
    
def dpo_train(model, tokenizer, dataset, dpo_save_path, config):
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
        gradient_accumulation_steps=config['dpo_gradient_steps'],
        learning_rate=config['dpo_lr'],
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

def utilization_offline_train(config, sft_train_flag = True, dpo_train_flag = True, re_generate_reference_data_flag = False):
    # Load Model
    sft_save_path = config['sft_save_path']
    dpo_save_path = config['dpo_save_path']
    if not os.path.exists(sft_save_path):
        os.mkdir(sft_save_path)
    if not os.path.exists(dpo_save_path):
        os.mkdir(dpo_save_path)
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if sft_train_flag:
        # Prepare SFT Dataset
        dataset = prepare_stf_dataset(tokenizer)

        # STF Training
        sft_train(model, tokenizer, dataset, sft_save_path, config)

    if dpo_train_flag:
        sft_checkpoint_path = config['sft_checkpoint_path']
        p_model = PeftModel.from_pretrained(model, model_id=sft_checkpoint_path)

        # Re-generate Reference Dataset
        if sft_train_flag or re_generate_reference_data_flag:
            generate_reference_data(model, tokenizer)

        # Prepare DPO Dataset
        dataset = prepare_dpo_dataset()
        # print(dataset)

        # DPO Training
        dpo_train(p_model, tokenizer, dataset, dpo_save_path, config)

def get_config():
    default_config = {
        'sft_checkpoint_path': '[Path for SFT checkpoint]',
        # SFT Config
        'sft_lr': 0.0001,
        'sft_gradient_steps': 16,
        # DPO Config
        'dpo_lr': 0.0001,
        'dpo_gradient_steps': 16,
        'sft_save_path': '[Path]',
        'dpo_save_path': '[Path]'
    }
    return default_config

if __name__ == '__main__':
    utilization_offline_train(
        get_config(),
        sft_train_flag=True,
        dpo_train_flag=True,
        re_generate_reference_data_flag=True
    )
