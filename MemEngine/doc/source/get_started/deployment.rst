Deployment
===============

There are two primary ways to use our library.

I. Local Deployment
-------------------

Developers can easily install our library in their Python environment via pip, conda, or from source code. Then, they can create memory modules for their agents, and utilize unified interfaces to perform memory operations within programs. An example is shown as follows:

.. code-block:: python

    from langchain.prompts import PromptTemplate
    from memengine.config.Config import MemoryConfig
    from memengine.memory.FUMemory import FUMemory
    ......

    class DialogueAgent():
        def __init__(self, role, another_role):
            self.llm = LLM()

            self.role = role
            self.another_role = another_role
            self.memory = FUMemory(MemoryConfig(DialogueAgentMemoryConfig))
        
        def response(self, observation):
            prompt = PromptTemplate(
                    input_variables=['role', 'memory_context', 'observation'],
                    template= DialogueAgentPrompt,
                ).format(role = self.role, memory_context = self.memory.recall(observation), observation = observation)
            res = self.llm.fast_run(prompt)
            self.memory.store('%s: %s\n%s: %s' % (self.another_role, observation, self.role, res))
            return res


More details can be found in :ref:`quick_start`.

II. Remote Deployment
---------------------

Alternatively, developers can install our library on computing servers and launch the service through a port.
First of all, you need to install ``uvicorn`` and ``fastapi`` as follows:

.. code-block:: bash

    pip install uvicorn fastapi


Then, lunch the service through a port with the following command:

.. code-block:: bash

    uvicorn server_start:memengine_server --reload --port [YOUR PORT]


Here, ``[YOUR PORT]`` is the port you provided such as ``8426``, and ``YOUR ADDRESS`` is the host address of the computing server.

Then, you can initiate a client to perform memory operations by sending HTTP requests remotely from their lightweight devices. An example is shown as follows:

.. code-block:: python
    
    from memengine.utils.Client import Client
    from langchain.prompts import PromptTemplate
    from memengine.config.Config import MemoryConfig
    from memengine.memory.FUMemory import FUMemory
    ......
    ServerAddress = 'http://127.0.0.1:[YOUR PORT]'

    class DialogueAgent():
        def __init__(self, role, another_role):
            self.llm = LLM()

            self.role = role
            self.another_role = another_role
            memory = Client(ServerAddress)
            memory.initilize_memory('FUMemory', DialogueAgentMemoryConfig)
        
        def response(self, observation):
            prompt = PromptTemplate(
                    input_variables=['role', 'memory_context', 'observation'],
                    template= DialogueAgentPrompt,
                ).format(role = self.role, memory_context = self.memory.recall(observation), observation = observation)
            res = self.llm.fast_run(prompt)
            self.memory.store('%s: %s\n%s: %s' % (self.another_role, observation, self.role, res))
            return res


You can also refer a complete example in ``run_client_sample.py``.