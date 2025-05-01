from default_config.DefaultGlobalConfig import *

# ----- LLM -----
DEFAULT_APILLM = {
    'method': 'APILLM',
    'name': 'gpt-4o-mini',
    'api_key': DEFAULT_OPENAI_APIKEY,
    'base_url': DEFAULT_OPENAI_APIBASE,
    'temperature': 0.9
}

# DEFAULT_LLM = DEFAULT_APILLM
DEFAULT_LLM = DEFAULT_QWEN_LOCAL

# ----- Truncation -----
DEFAULT_LMTRUNCATION = {
    # For truncation, you can choose two policy (word/token):
    # If the 'mode' is 'word', you just need provide the 'number' of words.
    # If the 'mode' is 'token', you need provide both the 'number' of words and the 'path' of tokenizer.
    'method': 'LMTruncation',
    'mode': 'word',
    'number': 8096,
    'path': DEFAULT_LLAMA3_8B_INSTRUCT_PATH
}

DEFAULT_TRUNCATION = DEFAULT_LMTRUNCATION

# ----- Encode -----
DEFAULT_LLAMA_ENCODER = {
    'method': 'LMEncoder',
    'name': 'Llama-3-8B-Instruct',
    'dimension': 4096,
    'path': DEFAULT_LLAMA3_8B_INSTRUCT_PATH
}

DEFAULT_E5_ENCODER = {
    'method': 'STEncoder',
    'name': 'e5-base-v2',
    'dimension': 768,
    'path': DEFAULT_E5_BASE_V2_PATH
}

# DEFAULT_ENCODER = DEFAULT_LLAMA_ENCODER
DEFAULT_ENCODER = DEFAULT_E5_ENCODER

# ----- Retrieval -----
DEFAULT_TEXT_RETRIEVAL = {
    # For retrieval, you can choose several policies.
    'method': 'TextRetrieval',
    'encoder': DEFAULT_ENCODER,
    'mode': 'cosine',
    'topk': 20,
}

DEFAULT_TIME_RETRIEVAL = {
    'method': 'TimeRetrieval',
    # For the argment (str) mode.
    # 'raw': No transform.
    'mode': 'raw',
    'topk': 20
}

DEFAULT_VALUE_RETRIEVAL = {
    'method': 'ValueRetrieval',
    'mode': 'identical',
    'topk': 20
}

# ----- Judge -----
DEFAULT_IMPORTANCE_JUDGE = {
    'method': 'LLMJudge',
    'LLM_config': DEFAULT_LLM,
    'prompt': {
        'template': """On the scale of 1 to 10, where 1 is purely unimportant and 10 is extremely important, rate the likely importance of the following piece of message.
Message: {message}
Your should just output the rating number between from 1 to 10, and do not output any other texts.""",
        'input_variables': ['message']
    },
    'post_scale': 10,
}

# ----- Reflector -----
DEFAULT_GAREFLECTOR = {
    'method': 'GAReflector',
    'LLM_config': DEFAULT_LLM,
    'question_prompt': {
        'template': """Information: {information}
Given only the information above, what are {question_number} most salient highlevel questions we can answer about the subjects in the statements?
Please output each question in a single line, and do not output any other messages.""",
        'input_variables': ['information', 'question_number']
    },
    'insight_prompt': {
        'template': """Statements: {statements}
What {insight_number} high-level insights can you infer from the above statements?
Please output each insight in a single line (without index), and do not output any other messages.""",
        'input_variables': ['statements', 'insight_number']
    },
    'question_number': 2,
    'insight_number': 2,
    'threshold': 0.3,
    'reflection_topk': 2,
}

DEFAULT_TRIALREFLECTOR = {
    'method': 'TrialReflector',
    'LLM_config': DEFAULT_LLM,
    'prompt': {
        'template': """Previous Insight: {previous_insight}
New Trial: {new_trial}
You are an advanced agent that can improve based on self-refection. You have been given a new trial in which you have interacted with the environment.
Based on the previous insight and the new trial, please generate a new insight text.
You may keep the information in the previous insight.
You may diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Your output should be consice, and do not output any other messages except for the new insight text.
Here is an example:
[Example]
{example}
You should just output the text of your new insight, without any other messages.""",
        'input_variables': ['previous_insight', 'new_trial', 'example']
    },
    'example': """Previous Insight: Alice likes to watch movies.
New Trial: (step 1) Alice: Could you recommend some movies to me? Assistant: Sure! What about Titanic and Before Sunrise? (Step 2) Alice: Great! I like this genre. Assistant: You are welcome!
Output: Alice likes to watch romantic movies."""

}

# ----- Summarizer -----
DEFAULT_LLMSUMMARIZER = {
    'method': 'LLMSummarizer',
    'LLM_config': DEFAULT_LLM,
    'prompt': {
        'template': """Content: {content}
Summarize the above content concisely, extracting the main themes and key information.
Please output your summary directly in a single line, and do not output any other messages.""",
        'input_variables': ['content']
    }
}

DEFAULT_SUMMARIZER = DEFAULT_LLMSUMMARIZER

# ----- Forget -----
DEFAULT_MBFORGET = {
    'method': 'MBForget',
    'coef': 5.0
}

DEFAULT_FORGET = DEFAULT_MBFORGET

# ----- Trigger -----
DEFAULT_MGMEMORY_LLMTRIGGER = {
    'LLM_config': DEFAULT_LLM,
    'func_list': [{
        'name': 'memory_retrieval',
        'args': ['query: str'],
        'args_type': ['str'],
        'func_description': 'retrieve query-related information from (external) archival storage, and add the result into working memory.',
        'args_description': ['query is a string to retrieve relevant information (e.g., \'Alice\'s name).']
    }, {
        'name': 'memory_recall',
        'args': ['query: str'],
        'args_type': ['str'],
        'func_description': 'retrieve query-related information from (external) recall storage, and add the result into FIFO memory.',
        'args_description': ['query is a string to retrieve relevant information (e.g., \'Alice\'s name).']
    }, {
        'name': 'memory_archive',
        'args': ['memory_list: list'],
        'args_type': ['list'],
        'func_description': 'archive some memory from FIFO memory into (external) archival storage.',
        'args_description': ['the index list of FIFO memory (e.g., [0, 2, 3]).']
    }, {
        'name': 'memory_transfer',
        'args': ['memory_list: list'],
        'args_type': ['list'],
        'func_description': 'transfer some memory from FIFO memory into working memory.',
        'args_description': ['the index list of FIFO memory (e.g., [0, 2, 3]).']
    }, {
        'name': 'memory_save',
        'args': ['memory_list: list'],
        'args_type': ['list'],
        'func_description': 'archive some memory from working memory into (external) archival storage.',
        'args_description': ['the index list of working memory (e.g., [0, 2, 3]).']
    }],
    'prompt': {
        'template': """Query: {text}
{memory_prompt}
{function_prompt}
Please choose some of the following functions to execute, and you should output each executed function in a single line.
{warning_content}{no_execute_prompt}Your output should follow the following format in the examples, and do not output other messages.
{few_shot}""",
        'input_variables': ['text', 'memory_prompt', 'function_prompt', 'warning_content', 'no_execute_prompt', 'few_shot']
    },
    'no_execute': 'If you do not execute any functions, just output [No Excuate].',
    'few_shot': """EXAMPLE 1:
memory_retrieval('Alice's name')
memory_save([0, 1])
EXAMPLE 2:
memory_recall('Bob's name')
memory_transfer([0])
EXAMPLE 3:
memory_archive([0])
EXAMPLE 4:
[No Excuate]"""
}

# ----- Utilization -----
DEFAULT_CONCATE_UTILIZATION = {
    'method': 'ConcateUtilization',
    'prefix': '[Memory Start]',
    'suffix': '[Memory End]',
    'list_config': {
        'index': True,
        'sep': '\n'
    },
    'dict_config': {
        'key_format': '(%s)',
        'key_value_sep': '\n',
        'item_sep': '\n'
    }
}

DEFAULT_UTILIZATION = DEFAULT_CONCATE_UTILIZATION