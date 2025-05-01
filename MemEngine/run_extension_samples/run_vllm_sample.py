import sys
sys.path.append('..')
from memengine import MemoryConfig
from memengine import GAMemory
from default_config.DefaultMemoryConfig import DEFAULT_GAMEMORY

# ----- Option 1: Use vllm with OpenAI Compatible Server -----
# You may start your vllm before using it.

# VllmAPIKey = 'EMPTY'
# VllmBaseURL = 'http://localhost:8613/v1'
# ModelPath = '/data/zhangzeyu/local_llms/Llama-3-8B-Instruct'

# DEFAULT_GAMEMORY['recall']['importance_judge']['LLM_config'] = {
#     'method': 'APILLM',
#     'name': ModelPath,
#     'api_key': VllmAPIKey,
#     'base_url': VllmBaseURL,
#     'temperature': 0.9
# }

# ----- Option 2: Use vllm with Local Models. -----
ModelPath = '/data/zhangzeyu/local_llms/Llama-3-8B-Instruct'
DEFAULT_GAMEMORY['recall']['importance_judge']['LLM_config'] = {
    'method': 'LocalVLLM',
    'name': ModelPath,
    'temperature': 0.9
}

def sample_vllm():
    memory_config = MemoryConfig(DEFAULT_GAMEMORY)
    memory = GAMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.manage('reflect')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

if __name__ == '__main__':
    sample_vllm()