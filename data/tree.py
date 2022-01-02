import os


class Tree:

    def __init__(self, key=None, val=None, children=None):
        self.key = key
        self.val = val
        self.children = {}

    def __str__(self):
        info = []
        def cb(tree, level): info.append(str(level) + ' : ' + tree.key)
        self.bfs(cb)
        return '\n'.join(info)

    def bfs(self, cb):
        q = [(self, 0)]
        i = 0
        while q and i < 20:
            i += 1
            curr, level = q.pop(0)
            cb(curr, level)
            for key, tree in curr.children.items():
                q.append((tree, level + 1))

    def put(self, key, val=None, children={}):
        assert key is not None
        keys = key.split(os.sep)
        if not self.key:
            self.key = keys[0]
        try:
            curr = self.children[keys[1]]
        except IndexError:
            self.key = key
            self.val = val
            self.children = children
            print('IE PUT ', key, ' IN ', self.key)
            return
        except KeyError:
            curr = Tree(keys[1])
            self.children[keys[1]] = curr
            print('KE PUT ', keys[1], ' IN ', self.key)
        curr.put(os.sep.join(keys[1:]), val=val, children=children)

    def get(self, key):
        assert key is not None
        if self.key == key:
            return self
        keys = key.split(os.sep)
        curr = self
        # for i in range(len(keys)):
        #     path = os.sep.join(keys[:i + 1])
        for k in keys:
            try:
                curr = curr.children[k]
            except KeyError:
                if curr.key == key:
                    return curr
                return None
        return curr

    def get_children(self, key=None):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        children = []

        def collect(tree, level):
            if not tree.children:
                children.append(tree)

        self.bfs(collect)
        return children

    def get_level(self, key=None, level=0):
        if not key:
            root = self
        else:
            root = self.get(key)
        assert root is not None
        trees = []

        def collect(tree, l):
            if l == level:
                trees.append(tree)

        self.bfs(collect)
        return trees


if __name__ == '__main__':

    path = 'test/split 1/class 1'

    tree = Tree()

    for root, dirs, files in os.walk(path):
        tree.put(root)
        if files:
            for f in files:
                tree.put(os.path.join(root, f))
                print(tree)
                print('')
        print(tree)
        print('')

    print(tree, '\n')

    target = tree.get('test/split 1')
    print(target)

    classes = tree.get_level(level=2)
    print([c.key for c in classes])

    children = target.get_children()
    print([c.key for c in children])
