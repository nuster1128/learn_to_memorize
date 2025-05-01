Memory Configurations
======================

In order to improve convenience for developers and facilitate parameter tuning by researchers, we have developed a unified memory configuration module.

- First, we design a hierarchical memory configuration module corresponding to our three-level memory implementations, enabling adjustments to both hyper-parameters and prompts within the memory models.
- Second, we provide a comprehensive set of default hyper-parameters and prompts, where developers and researchers can adjust only the specific parts without altering others.
- Finally, our configuration supports both statistic manners (\textit{e.g.,} files) and dynamic manners (\textit{e.g.,} dictionaries).