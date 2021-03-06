# Algorithms

## Table of contents

- [Tree Traversals](#tree-traversals)
    - [Preorder Traversal](#preorder-traversal)
    - [Inorder Traversal](#inorder-traversal)
    - [Postorder Traversal](#postorder-traversal)
    - [Level Traversal](#level-traversal)
    
- [Breadth First Search](#breadth-first-search)
- [Depth First Search](#depth-first-search)
- [Dijkstra algorithm](#dijkstra-algorithm)
- [Topological sorting](#topological-sorting)
- [Linked List](#linked-list)
- [Trie](#trie)
- [Binary Search](#binary-search)
- [Binary Search Rotated](#binary-search-rotated)

- [Prefix Sum](#prefix-sum)
- [Longest Common Subsequence](#longest-common-subsequence)
- [Longest Increasing subsequence](#longest-increasing-subsequence)
- [Levenshtein distance](#levenshtein-distance)
- [Catalan number](#catalan-number)
- [Matrix Multiplication](#matrix-multiplication)
- [Union Find](#union-find)
- [Segment Tree](#segment-tree)
- [Sparse Table](#sparse-table)
- [Bridges in a graph](#bridges-in-a-graph)
- [Bipartite Graph](#bipartite-graph)


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
#### c++
```python
#include <bits/stdc++.h>
using namespace std;


class Solution {
private:
    struct pair_hash {
    inline std::size_t operator()(const std::pair<int,int> & v) const {
        return v.first*31+v.second;
    }
};
public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int n = grid.size();
        int m = grid[0].size();
        int res = 0;
        unordered_set<pair<int, int>, pair_hash> visited;
        
        for (int i=0; i<n; i++) {
            for (int j=0; j<m; j++) {
                if (grid[i][j] && !visited.count({i,j})) {
                    visited.insert({i,j});
                    queue<pair<int, int>> qq;
                    qq.push(make_pair(i,j));
                    int tmp = 0;
                    while (!qq.empty()) {
                        tmp += 1;
                        auto node = qq.front();
                        qq.pop();
                        int a = node.first;
                        int b = node.second;
                        
                        vector<pair<int, int>> tmp = {{a+1,b},{a-1,b},{a,b+1},{a,b-1}};
                        for (auto [x,y]: tmp) {
                            if (0<=x && x<n && 0<=y && y<m && grid[x][y] && !visited.count({x,y})) {
                                visited.insert({x,y});
                                qq.push({x,y});
                            }
                        }
                    }
                    res = max(res, tmp);
                }
            }
        }
        
        return res;
    }
};
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

### Binary Search Rotated
```python
def bin_search_rotated(nums: List[int], target: int) -> int:
    start = 0
    end = len(nums) - 1
    while start <= end:
        mid = (start + end) // 2
        if nums[mid] == target:
            return mid
        if nums[start] <= target <= nums[mid]:
            end = mid - 1
        elif nums[mid] <= target <= nums[end]:
            start = mid + 1
        elif nums[end] <= nums[mid]:
            start = mid + 1
        elif nums[mid] <= nums[start]:
            end = mid - 1
        else:
            return -1
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
    grid = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
    for i in range(1, len(A) + 1):
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                grid[i][j] = 1 + grid[i - 1][j - 1]
            else:
                grid[i][j] = max(grid[i - 1][j], grid[i][j - 1])

    return grid[-1][-1]
```

### Longest Increasing subsequence
```python
from bisect import bisect_left


def lis(nums: List[int]) -> int:
    tmp = []
    for i in nums:
      x = bisect_left(tmp, i)
      if x == len(tmp):
        tmp.append(i)
      elif i < tmp[x]:
          tmp[x] = i
    
    return len(tmp)
```

```python
int lengthOfLIS(vector<int>& nums) {
        vector<int> res;
        for (auto x: nums) {
            int i = lower_bound(res.begin(), res.end(), x) - res.begin();
            if (i == res.size()) {
                res.push_back(x);
            }
            else {
                res[i] = x;
            }
        }
        
        return res.size();
    }
```

### Dijkstra algorithm
The main idea is use **heap** for storing costs for all nodes in not decreasing order. Also we heave **cost_visited** dictionary where we store
cost for arrived current node, it could be changed if we found way with less cost.
```python
from heapq import heappush, heappop


def dijkstra(grid: List[List[int]]) -> int:
    
    def get_neighbours(i, j):
        lst = []
        for x, y in [(i+x, j) for x in (-1, 1)] + [(i, j+y) for y in (-1, 1)]:
          if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            lst.append((grid[x][y], (x, y)))
        return lst
    
    adj_lst = {}
    
    for i in range(len(grid)):
          for j in range(len(grid[0])):
            adj_lst[(i, j)] = get_neighbours(i, j)
            
    start = (0, 0)
    goal = (len(grid)-1, len(grid[0])-1)
    heap = []
    heappush(heap, (0, start))
    cost_visited = {start: grid[0][0]}
    
    while heap:
            _, cur_node = heappop(heap)
            if cur_node == goal:
                break
            for neighbour in adj_lst[cur_node]:
                neigh_cost, neigh_node = neighbour
                actual_cost = neigh_cost + cost_visited[cur_node]
                if neigh_node not in cost_visited or actual_cost < cost_visited[neigh_node]:
                    cost_visited[neigh_node] = actual_cost
                    heappush(heap, (actual_cost, neigh_node))
                    
    return cost_visited[goal]
```

### Topological sorting
The node which hasn't any incomes calls _source_. The node which hasn't any outcomes call _sink_.
If we have nodes **U** and **V** and there is a route from **U** to **V** than **U** will be before than **V** in sorting.
First build _adj_lst_ and _income_count_ where key will be a node and value count of incomes. Then put into queue all sources.
If count of nodes in final sorting less than total nodes count it means that some nodes inside loops.

```python
from queue import Queue


def canFinish(num: int, prerequisites: List[List[int]]) -> bool:
        adj_lst, income_count = {}, {}
        
        for child, parent in prerequisites:
            adj_lst.setdefault(parent, []).append(child)
            income_count[child] = income_count.get(child, 0) + 1
            
        queue = Queue()
            
        for i in range(num):
            if i not in income_count:
                queue.put(i)
                
        topological_sorting = []
        
        while not queue.empty():
            node = queue.get()
            topological_sorting.append(node)
            for child in adj_lst.get(node, []):
                income_count[child] -= 1
                if not income_count[child]:
                    queue.put(child)
                    
        return num == len(topological_sorting)
```

### Trie
When we insert new word into **trie** we get structure like this 
```python
{'d': {'a': {'d': {'#': '#'}}}, 'p': {'a': {'d': {'#': '#'}}}}
```
There ar words
dad and pad. Here every symbol is a key and their value is a dict with possible next symbols and symbol **#** if it is the end of the word.

```python
class Trie:

    def __init__(self):
        self.trie = {}

    def insert(self, word: str) -> None:
        tmp = self.trie
        for ch in word:
            tmp = tmp.setdefault(ch, {})
        else:
            tmp["$"] = "$"

    def search(self, word: str) -> bool:
        tmp = self.trie
        for ch in word:
            tmp = tmp.get(ch)
            if not tmp:
                return False
        return "$" in tmp
```

### Catalan Number
```python
Cn = (2n)! / n!(n+1)!
```
```python
def catalan(n):
    nums = [0] * (n+1)
    nums[0] = nums[1] = 1
    for i in range(2, n+1):
        for j in range(i):
            nums[i] += nums[j]*nums[i-j-1];
            
    return nums[n]
```
#### c++
```python
unsigned long long catalan(int N)
{
    //used to store catalan numbers
    unsigned long long nums[N+1];

    nums[0]=nums[1]=1;
    int i,j;

    for(i=2;i<=N;i++)
    {
        nums[i]=0;
        for(j=0;j<i;j++)
            nums[i]+=nums[j]*nums[i-j-1];
    }
    return nums[N];
}
```

### Matrix multiplication
```python
def multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
      x, y, z = len(mat1), len(mat1[0]), len(mat2[0])
      grid = [[0] * z for _ in range(x)]
      for i in range(x):
          for j in range(z):
              for k in range(y):
                  grid[i][j] += mat1[i][k] * mat2[k][j]
      return grid
```

### Union Find
```python
class UnionFind:
    def __init__(self, size):
        self.root = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, x):
        if x == self.root[x]:
            return x
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

### Segment Tree

```python
class SegmentTree:

    def __init__(self, nums):
        n = len(nums)
        self.n = n if not (n & (n - 1)) else int("1" + "0" * (len(bin(n)) - 2), 2)
        self.segment_tree = [0] * (2 * self.n)
        for i in range(len(nums)):
            self.segment_tree[self.n + i] = nums[i]

        for i in range(self.n - 1, 0, -1):
            self.segment_tree[i] = self.segment_tree[i * 2] + self.segment_tree[i * 2 + 1]

    def update(self, index, value):
        index += self.n
        self.segment_tree[index] = value
        index //= 2
        while index:
            self.segment_tree[index] = self.segment_tree[2 * index] + self.segment_tree[2 * index + 1]
            index //= 2

    def query(self, left, right):
        left += self.n
        right += self.n
        res = 0
        while left <= right:
            if left % 2:
                res += self.segment_tree[left]
                left += 1
            if not right % 2:
                res += self.segment_tree[right]
                right -= 1

            left //= 2
            right //= 2

        return res
```

### Sparse Table

```python
class SparseTable:

    def __init__(self, nums: List):
        self.n = len(nums)
        self.log2 = [0] * (self.n+1)
        for i in range(2, self.n+1):
            self.log2[i] = self.log2[i//2] + 1
        pow_two = 1
        x = 2
        while x * 2 <= self.n:
            x *= 2
            pow_two += 1
        self.sp = [[float("inf")] * self.n for _ in range(pow_two + 1)]
        self.it = [[float("inf")] * self.n for _ in range(pow_two + 1)]
        for i in range(self.n):
            self.sp[0][i] = nums[i]
            self.it[0][i] = i

        for p in range(1, pow_two + 1):
            i = 0
            while i + (1 << p) <= self.n:
                left = self.sp[p - 1][i]
                right = self.sp[p - 1][i + (1 << (p - 1))]
                self.sp[p][i] = min(left, right)
                if left <= right:
                    self.it[p][i] = self.it[p - 1][i]
                else:
                    self.it[p][i] = self.it[p - 1][i + (1 << (p - 1))]
                
                i += 1

    def query_min(self, left: int, right: int):
        length = right - left + 1
        p = self.log2[length]
        k = 1 << p
        return min(self.sp[p][left], self.sp[p][right-k+1])

    def query_min_index(self, left, right):
        length = right - left + 1
        p = self.log2[length]
        k = 1 << p
        left_interval = self.sp[p][left]
        right_interval = self.sp[p][right - k + 1]
        if left_interval <= right_interval:
            return self.it[p][left]
        else:
            return self.it[right - p + 1]
```

### Bridges in a graph
```python
def find_bridges(n: int, connections: List[List[int]]) -> List[List[int]]:
  bridges = []
  adj_lst = {}
  visited = set()
  discovery_time = [float("inf")] * n
  # low - min id from neighbours ids and itself id
  low = [float("inf")] * n
  parent = [-1] * n
  time = 0
  for x, y in connections:
      adj_lst.setdefault(x, set()).add(y)
      adj_lst.setdefault(y, set()).add(x)

  def dfs(node, visited, parent, low, discovery_time):
      nonlocal time
      visited.add(node)
      discovery_time[node] = low[node] = time
      time += 1

      for nei in adj_lst.get(node, []):
          if nei not in visited:
              parent[nei] = node
              dfs(nei, visited, parent, low, discovery_time)

              low[node] = min(low[nei], low[node])

              # if nei_node doesn't connect to any node with less discover time than their edge is a bridge
              if low[nei] > discovery_time[node]:
                  bridges.append([node, nei])

          elif nei != parent[node]:
              low[node] = min(low[node], discovery_time[nei])

  for node in range(n):
      if node not in visited:
          dfs(node, visited, parent, low, discovery_time)

  return bridges
```

### Bipartite Graph
```python
from collections import deque


def isBipartite(graph: List[List[int]]) -> bool:
    n = len(graph)
    colors = {}
    for node in range(n):
        if node not in colors:
            colors[node] = False
            queue = deque()
            queue.append(node)
            while queue:
                node = queue.popleft()
                for nei in graph[node]:
                    if nei not in colors:
                        queue.append(nei)
                        colors[nei] = (not colors[node])
                    elif colors[nei] == colors[node]:
                        return False
    return True
```