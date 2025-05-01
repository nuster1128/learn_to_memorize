from MemEngine.default_config.DefaultMemoryConfig import *

score_config = {
    'moe_mode': 'trainable',
    'metrics_path': 'params/pretrain_params',
    'embedding_size': 768,
    'hidden_size': 256,
    'time_rank': 5,
}

RUS_DEFALUT_LLM = {
    'method': 'FastLLM',
    'name': 'Qwen2.5',
    'api_base': '[API_BASE]'
}

RUSMemoryConfig = {
    'name': 'RUSMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': {
        'method': 'RUSMemoryRecall',
        'truncation': DEFAULT_LMTRUNCATION,
        'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
        'time_retrieval': {
            'method': 'TimeRetrieval',
            'mode': 'delta'
        },
        'summarizer_with_hints': {
            'method': 'LLMSummarizer',
            'LLM_config': RUS_DEFALUT_LLM,
            'prompt': {
                'template': """Observation: {observation}
Hint: {hint}
From the above observation and according to the hint, please extract critical informative points and summarize them into a concise paragraph. You should just output the result of summarization, without any other messages.
""",
                'input_variables': ['observation', 'hint']
            }
        },
        'summarizer_without_hints': {
            'method': 'LLMSummarizer',
            'LLM_config': RUS_DEFALUT_LLM,
            'prompt': {
                'template': """Observation: {observation}
From the above observation, please extract critical informative points and summarize them into a concise paragraph. You should just output the result of summarization, without any other messages.
""",
                'input_variables': ['observation']
            }
        },
        'initial_hint': 'No hint.',
        'hint_usage': True,
        'score_config': score_config,
        'empty_memory': 'None',
        'utilization': {
            'aggregator_config':{
                'model_name': 'Qwen2.5-7B',
                'model_path': '[Model_Path]',
                'prompt': {
                    'template': """Observation: {observation}
Existing Memory: {memory_context}
New Memory: {new_memory}
Please merge the above new memory into the existing memory, which is useful to response the observation.
You should remove the duplicated information to make it concise, but do not lose any information.
You should just output the final memory after merge, without any other information.""",
                    'input_variables': ['observation', 'memory_context', 'new_memory']
                }
            },
        }
    },
    'store': {
        'method': 'RUSMemoryStore',
    },
    'online_train': {
        'train_epoch': 5,
        'sample_batch': 30,
        'save_dir': '[Your_PATH]',
        'chunk_size': 15,
        'sft_lr': 0.0005,
        'sft_gb': 16,
        'dpo_lr': 1e-4,
        'dpo_gb': 16,
    },
    # 'optimize': DEFAULT_RFMEMORY_OPTIMIZE,
    'display': DEFAULT_SCREEN_DISPLAY,
    'global_config': {
        'usable_gpu': '7'
    }
}