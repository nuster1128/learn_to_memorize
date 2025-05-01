Memory Functions
================

We implement various types of memory functions to support the construction of memory operations, which are listed as follows.

**Encoder** can transfer textual messages into embeddings to represent in latent space by pre-trained models, such as E5.

**Retrieval** is utilized to find most useful information for the current query or observation, commonly by different aspects like semantic relevance, importance, recency and so on.

**Reflector** aims to draw new insights in higher level from existing information, commonly for reflection and optimization operations.

**Summarizer** can summarize texts into a brief summary, which can decrease the lengths of texts and emphasize critical points.

**Trigger** is designed to call functions or tools in extensible manners. One significant instance is utilizing LLMs to determine which function should be called with specific arguments. 

**Utilization** aims to deal with several different parts of memory contents, formulating these information into a unified output.

**Forget** is typically applied in simulation-oriented agents, such role-playing and social simulations. It empowers agents with features of human cognitive psychology, aligning with human roles.

**Truncation** helps to formulate memory contexts under the limitations of token number by certain LLMs.

**Judge** intends to assess given observations or intermediate messages on certain aspects. For example, *GAMemory* judges the importance score of each observation when storing into memory, as an auxiliary criteria for the retrieval process.

**LLM** provides a convenient interface to utilize the powerful capability of different large language models.

All these memory functions are designed to conveniently construct different memory operations for various methods. For example, *GAMemoryStore* utilizes *LLMJudge* to provide the importance score on each observation.