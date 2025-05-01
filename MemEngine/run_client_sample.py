from memengine.utils.Client import Client
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY

ServerAddress = 'http://127.0.0.1:8426'

def sample_client():
    # Create a client to connect with the server.
    memory = Client(ServerAddress)
    # Initialize the memory model.
    memory.initilize_memory('FUMemory', DEFAULT_FUMEMORY)
    # Below is as same as the local usage.
    memory.reset()
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies?'))
    
if __name__ == "__main__":
    sample_client()