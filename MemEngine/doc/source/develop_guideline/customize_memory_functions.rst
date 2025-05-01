Customize Memory Functions
==============================

Researchers may need to implement new functions in their method. For example, they may extend *LLMJudge* to design a *BiasJudge* for poisoning detection. Here, we provide an example of *RandomJudge*:

.. code-block:: python

    from memengine.function import BaseJudge

    class MyBiasJudge(BaseJudge):
        def __init__(self, config):
            super().__init__(config)

        def __call__(self, text):
            return random.random()/self.config.scale
