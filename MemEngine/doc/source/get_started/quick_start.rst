.. _quick_start:

Quick Start
===============

We provide several manners to use MemEngine. We take local deployment as examples.


Using Stand-alone memory
------------------------

You can just run our sample `run_memory_samples.py` for the quick start.

.. code-block:: bash

    python run_memory_samples.py


Using memory in LLM-based agents
--------------------------------

We provide two example usage of applying MemEngine inside agents.

I. LLM-based Agents for HotPotQA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to install some dependencies as follows:

.. code-block:: bash

    pip install libzim beautifulsoup4


Then, download the wiki dump `wikipedia_en_all_nopic_2024-06.zim` and the data `hotpot_dev_fullwiki_v1.json` in your own path. After that, change the path and API keys in `cd run_agent_samples/run_hotpotqa.py`. And you can run the program with the command:

.. code-block:: bash

    cd run_agent_samples
    python run_hotpotqa.py


II. LLM-based Agents for Dialogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to change the API keys in `cd run_agent_samples/run_dialogue.py`. And you can run the program with the command:

.. code-block:: bash

    cd run_agent_samples
    python run_dialogue.py


Using memory with automatic selection
--------------------------------------

Developers can select the appropriate memory models, hyper-parameters, and prompts from the provided ranges, based on a specific task's criteria.

First of all, define a reward function as the ceriteria, whose input is a memory object and output is a float. An example of the dialogue task is shown as follows:

.. code-block:: python

    def sample_reward_func(memory):
        """Given a memory, utilize it and obtain a reward score to reflect how good it is.

        Args:
            memory (BaseMemory): the memory in MemEngine.

        Returns:
            float: the reward score to reflect how good the memory is.

        """
        dialogue_record = []

        user = DialogueAgent('User', 'Assistant', FUMemory(MemoryConfig(DEFAULT_FUMEMORY)))
        assistant = DialogueAgent('Assistant', 'User', memory)
        assistant_response = assistant.response('Please start the dialogue between User and Assistant.')

        for current_step in range(MAX_STEP):
            user_response = user.response(assistant_response)
            assistant_response = assistant.response(user_response)
            dialogue_record.append('User: %s' % user_response)
            dialogue_record.append('Assistant: %s' % assistant_response)

        score = eval_assistant(dialogue_record)
        return score

Then, prepare the range of model or config selection. An example is shown as follows:

.. code-block:: python

    # Option 1: Direct Assign
    ModelCandidate = [{
        'model': 'FUMemory',
        'config': DEFAULT_FUMEMORY
    },  {
        'model': 'LTMemory',
        'config': DEFAULT_LTMEMORY
    },  {
        'model': 'STMemory',
        'config': DEFAULT_STMEMORY
    }]

    # Option 2: Generate with Combination (Recommended for Hyper-parameter Tuning)
    ModelCandidate += generate_candidate({
        'model': 'LTMemory',
        'base_config': DEFAULT_LTMEMORY,
        'adjust_name': 'recall.text_retrieval.topk',
        'adjust_range': [1, 3, 5, 10]
    })

Finally, start automatic selection and get the result.

.. code-block:: python
    
    def sample_automode():
        selection_result = automatic_select(sample_reward_func, ModelCandidate)
        print('The full ranking of candidate is shown as follows:')
        print(selection_result)

        print('The best model/config is shown as follows:')
        print(selection_result[0])

The full example can be found in ``run_automode_sample.py``.
