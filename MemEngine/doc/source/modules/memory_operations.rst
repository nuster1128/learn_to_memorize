Memory Operations
=================

We implement various types of memory operations for constructing memory models, including store, recall, manage, and optimize.

Memory Store Operation
-----------------------

Intends to receive observations from the environment, processing them to obtain memory contents and adding them into memory storage. Another critical function of the memory store operation is to establish foundations for the memory recall operation, such as creating indexes and summaries.

Memory Recall Operation
-----------------------

Intends to obtain useful information to assist agents in their decision-making. Typically, the input is a query or observation representing the current state of agents. Some human-like agents may also endow the memory recall operation with certain retention characteristics like human memory.

Memory Manage Operation
-----------------------

Intends to reorganize existing information for better utilization, such as memory reflection. Besides, simulation-orientated agents may be equipped with a forgetting mechanism during the memory manage operation.

Memory Optimize Operation
-------------------------

Intends to optimize the memory capability of LLM-based agents by using extra trials and trajectories. It enables agents to extract meta-insight from historical experiences, which can be considered as a learn-to-memorize procedure.


Different memory models may share common memory operations or implement their unique operations according to their requirements. For example, *MTMemory* and *LTMemory* share the common memory recall operation *LTMemoryRecall*, while MTMemory has its own memory store operation *MTMemoryStore* to implement the tree-structured information update.
