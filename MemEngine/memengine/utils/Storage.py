from abc import ABC, abstractmethod
from memengine.function.Truncation import *

class BaseStorage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass

class LinearStorage(BaseStorage):
    """
    Memory storage in linear structure.
    """
    def __init__(self, config):
        super().__init__(config)

        self.memory_list = []
        self.counter = 0

    def reset(self):
        self.clear_memory()
        self.counter = 0

    def display(self):
        memory_display_items = []
        for m in self.memory_list:
            memory_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in m.items()]))
        if len(memory_display_items) == 0:
            return 'None'
        return '\n'.join(['[Memory Entity %d]\n%s' % (index, m) for index, m in enumerate(memory_display_items)])

    def clear_memory(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.get_element_number()
        
        assert 0 <= start <= end <= self.get_element_number()

        if start == 0:
            if end == self.get_element_number():
                self.memory_list = []
            else:
                self.memory_list = self.memory_list[end:]
        else:
            if end == self.get_element_number():
                self.memory_list = self.memory_list[:start]
            else:
                self.memory_list = self.memory_list[0:start] + self.memory_list[end:]

    def get_element_number(self):
        return len(self.memory_list)

    def is_empty(self):
        return self.get_element_number() == 0
    
    def get_memory_element_by_mid(self, mid):
        return self.memory_list[mid]

    def get_memory_attribute_by_mid(self, mid, attr):
        return self.memory_list[mid][attr]

    def get_memory_text_by_mid(self, mid):
        return self.get_memory_attribute_by_mid(mid, 'text')

    def get_mids_by_attribute(self, attr, value):
        mids = []
        for mid, m in enumerate(self.memory_list):
            if attr in m and m[attr] == value:
                mids.append(mid)
        return mids
    
    def update_memory_attribute_by_mid(self, mid, attr, value):
        self.memory_list[mid][attr] = value

    def add(self, obj):
        assert 'text' in obj
        obj['counter_id'] = self.counter
        self.memory_list.append(obj)
        self.counter += 1

    def delete_by_mid(self, mid):
        self.memory_list.pop(mid)

    def delete_by_mid_list(self, mids):
        for mid in sorted(mids, reverse=True):
            self.delete_by_mid(mid)

    def get_all_memory_in_order(self):
        return self.memory_list
    
class GraphStorage(BaseStorage):
    """
    Memory storage in graph structure.
    """
    def __init__(self, config):
        super().__init__(config)

        self.node = {}
        self.edge = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []

    def reset(self):
        self.node = {}
        self.edge = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []

    def display(self):
        node_display_items = []
        for node_id, element in self.node.items():
            node_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in element.items()]))
        if len(node_display_items) == 0:
            return 'None'
        
        edge_display_items = []
        for edge_id, element in self.edge.items():
            edge_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in element.items()]))

        if len(node_display_items) == 0:
            node_context =  'None'
        else:
            node_context = '\n'.join(['[Node Entity %d]\n%s' % (index, m) for index, m in enumerate(node_display_items)])
        
        if len(edge_display_items) == 0:
            edge_context =  'None'
        else:
            edge_context = '\n'.join(['[Edge Entity %d]\n%s' % (index, m) for index, m in enumerate(edge_display_items)])
        return """Memory Node Context::
%s
Memory Edge Context::
%s""" % (node_context, edge_context)
    
    def get_element_number(self):
        return len(self.node)

    def is_empty(self):
        return self.get_element_number() == 0
    
    def get_node_id_by_mid(self, mid):
        return self.memory_order_map[mid]

    def get_mid_by_node_id(self, node_id):
        return self.node[node_id]['mid']

    def get_memory_element_by_node_id(seld, node_id):
        return seld.node[node_id]

    def get_memory_element_by_mid(self, mid):
        node_id = self.get_node_id_by_mid(mid)
        return self.node[node_id]

    def get_memory_text_by_node_id(self, node_id):
        return self.node[node_id]['text']

    def get_memory_text_by_mid(self, mid):
        return self.get_memory_element_by_mid(mid)['text']

    def update_memory_attribute_by_node_id(self, node_id, attr, value):
        self.node[node_id][attr] = value
    
    def update_memory_attribute_by_mid(self, mid, attr, value):
        node_id = self.get_node_id_by_mid(mid)
        self.node[node_id][attr] = value

    def __update_memory_order_map__(self):
        node_id_list = list(self.node.keys())
        self.memory_order_map = node_id_list
        for mid, node_id in enumerate(node_id_list):
            self.node[node_id]['mid'] = mid

    def add_node(self, obj):
        assert 'text' in obj
        obj['node_id'] = self.node_counter
        obj['mid'] = len(self.memory_order_map)
        self.node[self.node_counter] = obj
        self.memory_order_map.append(self.node_counter)

        self.node_counter += 1
        return self.node_counter - 1

    def add_edge(self, s, t, obj):
        obj['edge_id'] = self.edge_counter
        if s not in self.edge:
            self.edge[s] = {}
        if t not in self.edge[s]:
            self.edge[s][t] = obj
        
        self.edge_counter += 1
        return self.edge_counter - 1
    
