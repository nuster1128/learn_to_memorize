from memengine import MemoryConfig

from memengine import FUMemory, STMemory, LTMemory, GAMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY, DEFAULT_LTMEMORY, DEFAULT_STMEMORY, DEFAULT_GAMEMORY

from memengine import MBMemory, SCMemory, MGMemory, RFMemory
from default_config.DefaultMemoryConfig import DEFAULT_MBMEMORY, DEFAULT_SCMEMORY, DEFAULT_MGMEMORY, DEFAULT_RFMEMORY

from memengine import MTMemory
from default_config.DefaultMemoryConfig import DEFAULT_MTMEMORY


def sample_FUMemory():
    memory_config = MemoryConfig(DEFAULT_FUMEMORY)
    memory = FUMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies?'))

def sample_STMemory():
    memory_config = MemoryConfig(DEFAULT_STMEMORY)
    memory = STMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies?'))

def sample_LTMemory():
    memory_config = MemoryConfig(DEFAULT_LTMEMORY)
    memory = LTMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies?'))

def sample_GAMemory():
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

def sample_MBMemory():
    memory_config = MemoryConfig(DEFAULT_MBMEMORY)
    memory = MBMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

def sample_SCMemory():
    memory_config = MemoryConfig(DEFAULT_SCMEMORY)
    memory = SCMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))
    print(memory.recall('Who is David?'))

def sample_MGMemory():
    memory_config = MemoryConfig(DEFAULT_MGMEMORY)
    memory = MGMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

def sample_RFMemory():
    memory_config = MemoryConfig(DEFAULT_RFMEMORY)
    memory = RFMemory(memory_config)
    trial1 = """Alice: I recently started a fascinating historical fiction book, and I can't put it down!
Assistant: What historical period does it cover?
Alice: It's set during the Renaissance, a time of incredible cultural and intellectual growth.
Assistant: The Renaissance sounds like such a vibrant era! How does the author weave historical facts into the story?"""
    memory.optimize(new_trial = trial1)
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

    trial2 = """Alice: I've been thinking about increasing my jogging distance. Do you have any tips for gradually improving endurance?
Assistant: Sure, Alice! You might want to start by adding an extra 5-10 minutes to your jog once a week to build stamina gradually.
Alice: I love jogging in the morning, but sometimes I feel a bit sluggish. Any advice to boost my energy before a run?
Assistant: Try having a light snack like a banana or some yogurt about 30 minutes before you head out. Staying hydrated also makes a big difference!"""
    memory.optimize(new_trial = trial2)
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    # memory.store('Alice loves reading and jogging.')
    memory.display()
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

def sample_MTMemory():
    memory_config = MemoryConfig(DEFAULT_MTMEMORY)
    memory = MTMemory(memory_config)
    memory.store('Alice is 28 years old and works as a university lecturer.')
    memory.store('Alice holds a master\'s degree in English Literature.')
    memory.display()
    memory.store('Alice loves reading and jogging.')
    memory.store('Alice has a pet cat named Whiskers.')
    memory.store('Last year, Alice traveled to New York to attend a literary conference.')
    memory.store('Bob is Alice\'s best friend, who is an excellent engineer.')
    print(memory.recall('What are Alice\'s hobbies in her spare time?'))

if __name__ == '__main__':
    sample_FUMemory()
    # sample_STMemory()
    # sample_LTMemory()
    # sample_GAMemory()
    # sample_MBMemory()
    # sample_SCMemory()
    # sample_MGMemory()
    # sample_RFMemory()
    # sample_MTMemory()