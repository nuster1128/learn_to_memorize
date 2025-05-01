from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *

DEFAULT_FUMEMORY = {
    'name': 'FUMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_STMEMORY = {
    'name': 'STMMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_STMEMORY_RECALL,
    'store': DEFAULT_STMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_LTMEMORY = {
    'name': 'LTMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_LTMEMORY_RECALL,
    'store': DEFAULT_LTMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_GAMEMORY = {
    'name': 'GAMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_GAMEMORY_RECALL,
    'store': DEFAULT_GAMEMORY_STORE,
    'reflect': DEFAULT_GAMEMORY_REFLECT,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_MBMEMORY = {
    'name': 'MBMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MBMEMORY_RECALL,
    'store': DEFAULT_MBMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_SCMEMORY = {
    'name': 'SCMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_SCMEMORY_RECALL,
    'store': DEFAULT_SCMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_MGMEMORY = {
    'name': 'MGMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MGMEMORY_RECALL,
    'store': DEFAULT_MGMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_RFMEMORY = {
    'name': 'RFMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'optimize': DEFAULT_RFMEMORY_OPTIMIZE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_MTMEMORY = {
    'name': 'MTMemory',
    'storage': DEFAULT_GRAPH_STORAGE,
    'recall': DEFAULT_LTMEMORY_RECALL,
    'store': DEFAULT_MTMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}


DEFAULT_ALL_PARAM = DEFAULT_FUMEMORY