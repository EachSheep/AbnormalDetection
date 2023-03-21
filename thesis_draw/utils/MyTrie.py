from collections import defaultdict
from graphviz import Digraph
from queue import Queue
# import marisa_trie as mt


class TrieNode(object):
    """字典树节点
    """

    def __init__(self, data):
        self.data = data
        self.children = defaultdict(TrieNode)
        self.node_count = 0  # 用于节点的数量统计
        self.edge_count = 0  # 用于边的流量统计
        self.ending = False  # 节点ending为True则表示当前节点存在数据

        self.id = ''  # 用于标识graphviz图中节点


class Trie(object):
    """字典树
    """

    def __init__(self):
        self.root = TrieNode('/')

    def insert_single(self, items):
        """往字典树里插入数据
        Args:
            items (list or str): 要插入trie树的数据
        """
        p = self.root
        for i, item in enumerate(items):
            if item not in p.children:
                p.children[item] = TrieNode(item)
                p.children[item].edge_count += 1
            else:
                p.children[item].edge_count += 1

            if i == len(items) - 1:
                p.children[item].node_count += 1

            p = p.children[item]

        p.ending = True

    def insert_multi(self, items, cnts):
        """往字典树里插入数据
        Args:
            items (list or str): 要插入trie树的数据
            cnts (list): 要插入trie树的数据对应的数量
        """
        p = self.root
        for i, item in enumerate(items):
            if item not in p.children:
                p.children[item] = TrieNode(item)
                p.children[item].edge_count += cnts[i]
            else:
                p.children[item].edge_count += cnts[i]

            if i == len(items) - 1:
                p.children[item].node_count += cnts[i]

            p = p.children[item]

        p.ending = True

    def find(self, items):
        """在Trie树中查找一个字符串/路径
        Args:
            items (list or str): 要查找的字符串/路径
        Returns:
            bool: True表示存在，False表示不存在
        """
        p = self.root
        for item in items:
            if item not in p.children:
                return False
            p = p.children[item]
        if p.ending:
            return True
        else:
            return False

    def draw_trie(self):
        """层序遍历绘图
        """
        if self.root == None:
            return None

        g = Digraph('Trie', node_attr={'color': 'lightblue2'})

        traverse_res = []  # 存放层序遍历结果，用于后面连接graphviz图中的节点
        q = Queue(maxsize=-1)
        q.put(self.root)
        level_id, col_id = 'A', 0  # level_id: 第几层, col_id: 某层中节点的索引, level与idx一起组成trie树节点的id
        while not q.empty():
            cur_len = q._qsize()
            cur_level_nodes = []
            for i in range(cur_len):
                id = '{}{}'.format(level_id, col_id)  # 节点id
                node = q.get()
                node.id = id
                cur_level_nodes.append(node)
                g.node(name=id, label=node.data + "-" + str(node.node_count))  # 绘制一个节点

                for key, item in node.children.items():
                    q.put(item)
                
                col_id += 1  # idx标记向后移动一位
            level_id = chr(ord(level_id) + 1)  # 进入下一层
            traverse_res.append(cur_level_nodes)

        #  绘制边
        for level_nodes in traverse_res:
            for node in level_nodes:
                for key, child_node in node.children.items():
                    g.edge(node.id, child_node.id, str(child_node.edge_count))
        
        # g.view()
        return g


if __name__ == '__main__':

    paths = [
        '/Users/xxxx/Documents/lll.mp4', 
        '/Applications/WeChat.app/Contents/Resources/Compose_Video.svg',
        '/Applications/WeChat.app/Contents/Resources/Compose_Collapsed.svg', 
        '/private/var/folders/xxxx'
    ]
    trie_data = [path[1:].split('/') for path in paths]
    tree = Trie()
    for path in trie_data:
        tree.insert_single(path)
    print(tree.find(['Users', 'xxxx', 'Documents', 'lll.mp4']))  # True
    print(tree.find(['Users', 'xxxx', 'Documents']))

    g = tree.draw_trie()
    g.view(filename = 'Trie', directory = './', cleanup = True)