# Algorithms

## Table of contents

- [Tree Traversals](#tree-traversals)
    - [Preorder Traversal](#preorder-traversal)
    - [Inorder Traversal](#inorder-traversal)
    - [Postorder Traversal](#postorder-traversal)
    - [Level Traversal](#level-traversal)
    
- [Breadth First Search](#breadth-first-search)
- [Depth First Search](#dfs)
- [Dijkstra algorithm](#dijkstra)
- [Linked List](#linked_list)
- [Trie](#trie)
- [Binary Search](#bin_search)
    - [Binary Search(Rotated)](#bin_search_rotated)
    
- [Prefix Sum](#prefix_sum)


### Tree Traversals
Ways to collect tree nodes(values)

```python
class TreeNode:
  def __init__(self):
    self.val = 0
    self.left = None
    self.right = None
```

#### Preorder Traversal
```python
def preorder_traversals(root: TreeNode) -> List[int]:
    result = []
    
    def _preorder(root):
      if not root:
        return None
      result.append(root.val)
      if root.left:
        _preorder(root.left)
      if root.right:
        _preorder(root.right)
        
    _preorder(root)
    
    return result
```

#### Inorder Traversal
```python
def inorder_traversals(root: TreeNode) -> List[int]:
    result = []
    
    def _inorder(root):
      if not root:
        return None
      if root.left:
        _inorder(root.left)
      result.append(root.val)
      if root.right:
        _inorder(root.right)
        
    _inorder(root)
    
    return result
```

#### Postorder Traversal
```python
def postorder_traversals(root: TreeNode) -> List[int]:
    result = []
    
    def _postorder(root):
      if not root:
        return None
      if root.left:
        _postorder(root.left)
      if root.right:
        _postorder(root.right)
      result.append(root.val)
        
    _postorder(root)
    
    return result
```

#### Level Traversal
```python
from queue import Queue


def level_traversal(root) -> List[List[int]]:
  result = []
  queue = Queue()
  queue.put(root)
  
  while not queue.empty():
    level = []
    size = queue.qsize()
    while size:
      node = queue.get()
      level.append(node.val)
      if node.left:
        queue.put(node.left)
      if node.right:
        queue.put(node.right)
      size -= 1
    else:
      result.append(level)
      
  return result
```

### Breadth First Search
The idea is exploring the nearest neighbours using `queue`. Good tool for searching shortest path on undirected and
unweighted graph.
```python
from queue import Queue


def bfs(grid: List[List[int]]) -> int:
    
    def get_neighbours(i, j):
        lst = []
        for x, y in [(i+x, j) for x in (-1, 1)] + [(i, j+y) for y in (-1, 1)]:
          if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            lst.append((x, y))
        return lst
    
    adj_lst = {}
    steps = {}
    
    for i in range(len(grid)):
      for j in range(len(grid[0])):
        adj_lst[(i, j)] = get_neighbours(i, j)
        steps[(i, j)] = -1
        
    queue = Queue()
    start = (0, 0)
    end = (len(grid)-1, len(grid[0])-1)
    queue.put(start)
    visited = set()
    steps[start] = 0
    
    while not queue.empty():
      node = queue.get()
      for nei in adj_lst[node]:
        if nei not in visited:
          steps[nei] = steps[node] + 1
          visited.add(nei)
          queue.put(nei)
          
    return steps[end]

```