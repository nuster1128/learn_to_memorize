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

HotpotQAConfig = {
    # Configurations for HotpotQA Environment.
    'environment_config': {
        'dataset_path': './dataset',
        'max_try': 5,
        'wiki_path': '[PATH for wikipedia_en_all_nopic_2024-06.zim]'
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
    
def run_hotpot_qa(config, result_path_suffix = ''):
    data_generate_index = 0
    result_path = 'offline_train/%s-%s-%d.jsonl' % (config['agent_config']['llm_config']['name'], config['agent_config']['memory_config']['method'], data_generate_index)

    environment = HotPotQAEnv(config['environment_config'])
    agent = HotPotQAAgent(config['agent_config'])

    bias = 0
    generate_start = 1000 * (data_generate_index) + bias
    generate_total_num = 1000

    environment.current_train_id += generate_start
    total_score = 0
    for i in range(generate_total_num):
        print('----- Generate %d Start -----' % i)
        agent.reset()
        observation, reward, terminated, info = environment.reset_train()
        logging.info('Environment Observation (%s): %s' % (info['step_id'], observation))

        step_info_list = []
        while not terminated:
            action, step_info = agent.response(observation, reward, terminated, info)
            step_info_list.append(step_info)
            logging.info('Agent Action (%s): %s' % (info['step_id'],action))
            observation, reward, terminated, info = environment.step(action)
            logging.info('Environment Observation (%s): %s' % (info['step_id'], observation))
        logging.info('[Trjectory %d] Final result: %d.' % (i,reward))
        total_score += reward
        logging.info('[Trjectory %d] Total Score: %d.' % (i,total_score))
        trial_info = {
            'task_id': environment.current_train_id,
            'task_info': environment.task_info,
            'step_info_list': step_info_list,
            'total_step': info['step_id'],
            'reward': reward
        }
        with open(result_path, 'a', encoding='utf-8') as f:
            json_line = json.dumps(trial_info,ensure_ascii=False)
            f.write(json_line + '\n')
        
    print('Total Score: %d' % total_score)

if __name__ == '__main__':
    run_hotpot_qa(config=HotpotQAConfig)