Memory Models
===============

We implement a variety of memory models from recent research works under a general structure, allowing seamless switching among them. Specifically, these models are implemented with the interfaces including reset, store, recall, manage, and optimize.

Our implemented memory models are shown as follows:

- **FUMemory** (Full Memory): Naively concatenate all the information into one string, also known as long-context memory.
- **LTMemory** (Long-term Memory): Calculate semantic similarities with text embeddings to retrieval most relevant information.
- **STMemory** (Short-term Memory): Maintain the most recent information and concatenate them into one string as the context.
- **GAMemory** (Generative Agents [1]): A pioneer memory model with weighted retrieval combination and self-reflection mechanism.
- **MBMemory** (MemoryBank [2]): A multi-layered memory model with dynamic summarization and forgetting mechanism.
- **SCMemory** (SCM [3]): A self-controlled memory model that can recall minimum but necessary information for inference.
- **MGMemory** (MemGPT [4]): A hierarchical memory model that treat the memory system as an operation system.
- **RFMemory** (Reflexion [5]): A famous memory method that can learn to memorize from previous trajectories by optimization.
- **MTMemory** (MemTree [6]): A dynamic memory model with a tree-structured semantic representation to organize information.


All of these memory models are implemented with the combination among various memory operations, and we make some reasonable adaptations in their implementations.

References
----------

[1] Park, Joon Sung, et al. "Generative agents: Interactive simulacra of human behavior." Proceedings of the 36th annual acm symposium on user interface software and technology. 2023.

[2] Zhong, Wanjun, et al. "Memorybank: Enhancing large language models with long-term memory." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.

[3] Wang, Bing, et al. "Enhancing large language model with self-controlled memory framework." arXiv preprint arXiv:2304.13343 (2023).

[4] Packer, Charles, et al. "Memgpt: Towards llms as operating systems." arXiv preprint arXiv:2310.08560 (2023).

[5] Shinn, Noah, et al. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).

[6] Rezazadeh, Alireza, et al. "From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs." arXiv preprint arXiv:2410.14052 (2024).