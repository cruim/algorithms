# Algorithms

## Table of contents

- [Tree Traversals](#tree-traversals)
    - [Preorder Traversal](#preorder-traversal)
    - [Inorder Traversal](#inorder-traversal)
    - [Postorder Traversal](#postorder-traversal)
    - [Level Traversal](#level-traversal)
    
- [Breadth First Search](#breadth-first-search)
- [Depth First Search](#depth-first-search)
- [Dijkstra algorithm](#dijkstra)
- [Linked List](#linked-list)
- [Trie](#trie)
- [Binary Search](#binary-search)
    - [Binary Search(Rotated)](#binary-search-rotated)
    
- [Prefix Sum](#prefix-sum)
- [Longest Common Subsequence](#longest-common-subsequence)
- [Longest Increasing subsequence](#longest-increasing-subsequence)
- [Levenshtein distance](#levenshtein-distance)


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

### Depth First Search
The idea is to get in destination point as soon as possible using `stack`. Good tool for collecting all paths. 
```python
def dfs(grid: List[List[int]]) -> List:
    def get_neighbours(i, j):
        lst = []
        for x, y in [(i+x, j) for x in (-1, 1)] + [(i, j+y) for y in (-1, 1)]:
          if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            lst.append((x, y))
        return lst
    
    adj_lst = {}
    
    for i in range(len(grid)):
          for j in range(len(grid[0])):
            adj_lst[(i, j)] = get_neighbours(i, j)
    
    start = (0, 0)
    end = (len(grid)-1, len(grid[0])-1)
    visited = set()
    result = []
    path = []

    def _dfs(start, end, visited, path):
        path.append(start)
        visited.add(start)
        if start == end:
          result.append(path.copy())
        else:
          for nei in adj_lst[start]:
            if nei not in visited:
              _dfs(nei, end, visited, path)
              
        path.pop()
        visited.discard(start)
        
    _dfs(start, end, visited, path)
    
    return result
```

### Binary Search
Searching on sorted array, `O(log(n))`
```python
def binary_search(nums, value):
    low = 0
    high = len(nums) - 1
    while low <= high:
      mid = (low + high) // 2
      if nums[mid] == value:
        return mid
      elif value < nums[mid]:
        high = mid - 1
      else:
        low = mid + 1
        
    return -1
```

### Levenshtein distance
Using to find minimum numbers of edits to make first string from second. Available edits:
_(Replace character, Insert character, Remove character)_
```python
def levenshtein_distance(A: str, B: str) -> int:
    grid = [[(i+j) if not i*j else 0 for j in range(len(B)+1)] for i in range(len(A)+1)]
    for i in range(1, len(A)+1):
      for j in range(1, len(B)+1):
        if A[i-1] == B[j-1]:
          grid[i][j] = grid[i-1][j-1]
        else:
          grid[i][j] = 1 + min(grid[i-1][j], grid[i][j-1], grid[i-1][j-1])
          
    return grid[len(A)][len(B)]
```

### Longest Common Subsequence
```python
def lsc(A: str, B: str) -> int:
    grid = [[0] * (len(B)+1) for _ in range(len(A)+1)]
    for i in range(1, len(A)+1):
      for j in range(len(B)+1):
        if A[i-1] == B[j-1]:
          grid[i][j] = 1 + grid[i-1][j-1]
        else:
          grid[i][j] = max(grid[i-1][j], grid[i][j-1])
          
    return grid[-1][-1]
```
