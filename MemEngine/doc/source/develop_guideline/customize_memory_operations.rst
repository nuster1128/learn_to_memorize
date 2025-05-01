Customize Memory Operations
==============================
To implement a new method, the memory operation is most significant part to customize, containing major pipelines of the detailed process. Here is an example:

.. code-block:: python

    ......

    class MyMemoryRecall(BaseRecall):
        def __init__(self, config, **kwargs):
            super().__init__(config)

            self.storage = kwargs['storage']
            self.insight = kwargs['insight']
            self.truncation = LMTruncation(self.config.truncation)
            self.utilization = ConcateUtilization(self.config.utilization)
            self.text_retrieval = TextRetrieval(self.config.text_retrieval)
            self.bias_retrieval = ValueRetrieval(self.config.bias_retrieval)
        
        def reset(self):
            self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.bias_retrieval])
        
        @__recall_convert_str_to_observation__
        def __call__(self, query):
            if self.storage.is_empty():
                return self.config.empty_memory
            text = query['text']
            
            relevance_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
            bias, _ = self.bias_retrieval(None, topk=False, with_score = True, sort = False)
            final_scores = relevance_scores + bias
            scores, ranking_ids = torch.sort(final_scores, descending=True)

            if hasattr(self.config, 'topk'):
                scores, ranking_ids = scores[:self.config.topk], ranking_ids[:self.config.topk]

            memory_context = self.utilization({
                        'Insight': self.insight['global_insight'],
                        'Memory': [self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids]
                    })

            return self.truncation(memory_context)