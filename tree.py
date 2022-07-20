class Node:
    def __init__(self, data, condition):
        self.data = data
        self.condition = condition
        self.answer = ""
        self.left = None
        self.right = None

class Tree:
    def __init__(self, data):
        node = Node(data, "")
        self.root = node


