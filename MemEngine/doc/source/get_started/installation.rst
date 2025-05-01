Installation
===============
There are several ways to install MemEngine. We recommend the environment version with ``python>=3.9``.

I. Install from source code (Recommended)
------------------------------------------

We highly recommend installing MemEngine from source code.

.. code-block:: bash

    conda create -n memengine_env python=3.9
    git clone https://github.com/nuster1128/MemEngine.git
    cd MemEngine
    pip install -e .


II. Install from pip
--------------------

Developers may also install MemEngine with ``pip``, but it might not be the latest version.

.. code-block:: bash

    conda create -n memengine_env python=3.9
    pip install memengine

III. Install from conda
-----------------------

Developers can install MemEngine from conda. When installing MemEngine from conda, please add `conda-forge` into your channel to ensure langchain can be installed properly.

.. code-block:: bash
    
    conda create -n memengine_env python=3.9
    conda install memengine