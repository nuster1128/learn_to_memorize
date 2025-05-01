.. title:: MemEngine v1.0.2

MemEngine
===========
Introduction
------------

Many research methods have been proposed to improve the memory capability of LLM-based agents, however, they are implemented under different pipelines and lack a unified framework.
This inconsistency presents challenges for developers to attempt different models in their experiments.Moreover, many basic functions (such as retrieval) are duplicated across different models, and researchers often need to implement them repeatedly when developing new models, leading to wasted time.
Besides, many academic models are tightly integrated with agents in a non-pluggable manner, making them difficult to apply across different agents.

.. image:: asset/framework.png


Features
---------

**Unified and Modular Memory Framework.** We propose a unified memory framework composed of three hierarchical levels to organize and implement existing research models under a general structure. All these three levels are modularized inside our framework, where higher-level modules can reuse lower-level modules, thereby improving implementation efficiency and consistency. Besides, we provide a configuration module for easy modification of hyper-parameters and prompts at different levels, and implement a utility module to better save and demonstrate memory contents.

**Abundant Memory Implementation.** Based on our unified and modular framework, we implement a wide range of memory models from recent research works, many of which are widely applied in diverse applications. All of these models can be easily switched and tested under our framework, with different configurations of hyper-parameters and prompts that can be adjusted for better application across various agents and tasks.

**Convenient and Extensible Memory Development.** Based on our modular memory operations and memory functions, researchers can conveniently develop their own advanced memory models. They can also extend existing operations and functions to develop their own modules. To better support researchers' development, we provide detailed instructions and examples in our document to guide the customization.

**User-friendly and Pluggable Memory Usage.** Our library offers multiple deployment options, and provides various memory usage modes, including default, configurable, and automatic modes. Moreover, our memory modules are pluggable and can be easily utilized across different agent framework, which is also compatible with some prominent frameworks.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/deployment
   get_started/quick_start
   get_started/versions

.. toctree::
   :maxdepth: 1
   :caption: Modules

   modules/overview
   modules/memory_methods
   modules/memory_operations
   modules/memory_functions
   modules/memory_configs
   modules/memory_utils

.. toctree::
   :maxdepth: 1
   :caption: Develop Guideline

   develop_guideline/customize_memory_methods
   develop_guideline/customize_memory_operations
   develop_guideline/customize_memory_functions

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_references/TBD.rst

Acknowledge
-----------