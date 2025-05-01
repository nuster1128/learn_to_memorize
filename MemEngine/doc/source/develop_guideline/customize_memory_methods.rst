Customize Memory Methods
==============================

By utilizing the newly customized memory operations and the existing ones, research can formulate their methods with various combinations in final. Here is an example:

.. code-block:: python

    ......

    class MyMemory(ExplicitMemory):
        def __init__(self, config) -> None:
            super().__init__(config)
            
            self.storage = LinearStorage(self.config.args.storage)
            self.insight = {'global_insight': '[None]'}

            self.recall_op = MyMemoryRecall(
                self.config.args.recall,
                storage = self.storage,
                insight = self.insight
            )
            self.store_op = MyMemoryStore(
                self.config.args.store,
                storage = self.storage,
                text_retrieval = self.recall_op.text_retrieval,
                bias_retrieval = self.recall_op.bias_retrieval
            )
            self.optimize_op = RFOptimize(self.config.args.optimize, insight = self.insight)

            self.auto_display = ScreenDisplay(self.config.args.display, register_dict = {
                'Memory Storage': self.storage,
                'Insight': self.insight
            })

        def reset(self):
            self.__reset_objects__([self.storage, self.store_op, self.recall_op])
            self.insight = {'global_insight': '[None]'}

        def store(self, observation) -> None:
            self.store_op(observation)
        
        def recall(self, observation) -> object:
            return self.recall_op(observation)

        ......


The full example can be found in `run_custom_samples.py`.