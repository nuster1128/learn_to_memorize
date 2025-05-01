import sys, logging,json
sys.path.append('learn_to_memorize_project')
sys.path.append('learn_to_memorize_project/MemEngine')
from experiments.hotpot_qa_agent import HotPotQAAgent
from experiments.hotpot_qa_env import HotPotQAEnv
from experiments.rus_config import RUSMemoryConfig
from experiments.RUSMemory import ImportanceScorer, EmotionScorer, MoEGate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

OPENAI_MODEL = 'gpt-4o-mini'
OPENAI_APIKEY = '[API_KEY]'
OPENAI_APIBASE = '[API_BASE]'

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

HotpotQAConfig = {
    # Configurations for HotpotQA Environment.
    'environment_config': {
        'dataset_path': './dataset',
        'level': 'hard',
        'max_try': 5,
        'wiki_path': '[PATH of wikipedia_en_all_nopic_2024-06.zim]'
    },
    # Configurations for HotpotQA Environment.
    'agent_config': {
        'llm_config': {
            'name': OPENAI_MODEL,
            'api_key': OPENAI_APIKEY,
            'api_base': OPENAI_APIBASE
        },
        'memory_config': {
            'method': 'RUSMemory',
            'result_name': 'RUSMemory',
            'config': RUSMemoryConfig
        }
    }
}

# Load offline trained offline learning.
# HotpotQAConfig['agent_config']['memory_config']['config']['recall']['score_config']['params_path'] = 'params/offline_trained_params/trained_moe_gate.pickle'
# HotpotQAConfig['agent_config']['memory_config']['config']['recall']['utilization']['aggregator_config']['sft_model_path'] = 'params/offline_trained_params/utilization_checkpoints_SFT/checkpoint-225'
# HotpotQAConfig['agent_config']['memory_config']['config']['recall']['utilization']['aggregator_config']['dpo_model_path'] = 'params/offline_trained_params/utilization_checkpoints_DPO/checkpoint-112'
# HotpotQAConfig['agent_config']['memory_config']['config']['recall']['initial_hint'] = load_json('params/offline_trained_params/hint.json')['hint']

# If conduct online training, make it true.
Online_Train_Flag = True

    
def run_hotpot_qa(config, result_path_suffix = ''):
    if 'result_name' in config['agent_config']['memory_config']:
        result_name = config['agent_config']['memory_config']['result_name']
    else:
        result_name = config['agent_config']['memory_config']['method']
    result_path = 'results/%s-%s%s.jsonl' % (config['agent_config']['llm_config']['name'], result_name, result_path_suffix)

    environment = HotPotQAEnv(config['environment_config'])
    agent = HotPotQAAgent(config['agent_config'])

    # Joint Training Phase
    if Online_Train_Flag:
        agent.online_train(environment)
        raise

    if HotpotQAConfig['environment_config']['level'] == 'hard':
        total_eval_num = 113
    elif HotpotQAConfig['environment_config']['level'] == 'medium':
        total_eval_num = 109
    elif HotpotQAConfig['environment_config']['level'] == 'easy':
        total_eval_num = 107
    else:
        raise
    
    # Manually Load Checkpoint
    start = 0
    environment.current_test_id = start

    # total_eval_num = environment.get_number_of_test_task()[1]
    # Evaluation Phase
    total_score = 0
    for i in range(start, total_eval_num):
        print('----- Eval %d Start -----' % i)
        agent.reset()
        observation, reward, terminated, info = environment.reset_test()
        logging.info('[system] Environment Observation (%s): %s' % (info['step_id'], observation))

        step_info_list = []
        while not terminated:
            action, step_info = agent.response(observation, reward, terminated, info)
            step_info_list.append(step_info)
            logging.info('[system] Agent Action (%s): %s' % (info['step_id'],action))
            observation, reward, terminated, info = environment.step(action)
            logging.info('[system] Environment Observation (%s): %s' % (info['step_id'], observation))
        agent.response(observation, reward, terminated, info)
        logging.info('[system] [Trjectory %d] Final result: %d.' % (i,reward))
        total_score += reward
        logging.info('[system] [Trjectory %d] Total Score: %d.' % (i,total_score))
        trial_info = {
            'eval_id': i,
            'task_info': environment.task_info,
            'last_action': action,
            'total_step': info['step_id'],
            'reward': reward
        }
        with open(result_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(trial_info,ensure_ascii=False)
            f.write(json_line + '\n')
        
    print('Total Score: %d' % total_score)

if __name__ == '__main__':
    run_hotpot_qa(config=HotpotQAConfig)