"""
图基础概念复习 —— 从 Java 到 Python
==================================
核心概念：
  - 顶点 (Vertex / Node)
  - 边 (Edge)：有向 / 无向，加权 / 无权
  - 邻接表 (Adjacency List)  vs  邻接矩阵 (Adjacency Matrix)
  - 度 (Degree)：入度 (In-degree) / 出度 (Out-degree)
"""

# ============================================================
# 1. 邻接表实现（类比 Java 的 ArrayList<Integer>[] 或 Map）
# ============================================================
class GraphAdjList:
    """无向无权图 —— 邻接表"""

    def __init__(self):
        self.adj = {}  # dict: node -> list[neighbor]

    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u, v):
        """添加无向边"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        self.adj[v].append(u)

    def degree(self, v):
        return len(self.adj.get(v, []))

    def neighbors(self, v):
        return self.adj.get(v, [])

    def __str__(self):
        return "\n".join(f"{v}: {neighbors}" for v, neighbors in self.adj.items())


# ============================================================
# 2. 邻接矩阵实现
# ============================================================
class GraphAdjMatrix:
    """无向无权图 —— 邻接矩阵"""

    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v):
        if 0 <= u < self.n and 0 <= v < self.n:
            self.matrix[u][v] = 1
            self.matrix[v][u] = 1  # 无向

    def degree(self, v):
        return sum(self.matrix[v])

    def neighbors(self, v):
        return [i for i in range(self.n) if self.matrix[v][i] == 1]

    def __str__(self):
        header = "  " + " ".join(str(i) for i in range(self.n))
        rows = [header]
        for i, row in enumerate(self.matrix):
            rows.append(f"{i}: {row}")
        return "\n".join(rows)


# ============================================================
# 3. 有向图（带权）
# ============================================================
class DiGraph:
    """有向加权图"""

    def __init__(self):
        self.adj = {}  # node -> list of (neighbor, weight)

    def add_edge(self, u, v, weight=1):
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append((v, weight))

    def out_degree(self, v):
        return len(self.adj.get(v, []))

    def in_degree(self, v):
        count = 0
        for u in self.adj:
            for w, _ in self.adj[u]:
                if w == v:
                    count += 1
        return count


# ============================================================
# 4. 遍历：DFS & BFS
# ============================================================
def dfs(graph: GraphAdjList, start):
    """深度优先遍历"""
    visited = set()
    result = []

    def _dfs(v):
        visited.add(v)
        result.append(v)
        for w in graph.neighbors(v):
            if w not in visited:
                _dfs(w)

    _dfs(start)
    return result


def bfs(graph: GraphAdjList, start):
    """广度优先遍历"""
    from collections import deque

    visited = {start}
    q = deque([start])
    result = []

    while q:
        v = q.popleft()
        result.append(v)
        for w in graph.neighbors(v):
            if w not in visited:
                visited.add(w)
                q.append(w)

    return result


# ============================================================
# 5. Dijkstra 最短路径
# ============================================================
import heapq


def dijkstra(graph: DiGraph, start):
    """
    Dijkstra 算法 —— 单源最短路径。

    核心思想：贪心 + 松弛 (relaxation)
      - 每次从未处理的节点中选距离最小的
      - 通过它尝试"松弛"邻居的距离
      这就是 BFS 在有权图上的推广 —— 用优先队列代替普通队列
    """
    INF = float('inf')
    dist = {v: INF for v in graph.adj}
    prev = {v: None for v in graph.adj}   # 记录路径
    dist[start] = 0

    # 优先队列: (当前距离, 节点)
    pq = [(0, start)]
    visited = set()

    while pq:
        d, v = heapq.heappop(pq)

        if v in visited:
            continue
        visited.add(v)

        for w, weight in graph.adj[v]:
            if w in visited:
                continue

            new_dist = d + weight
            if new_dist < dist[w]:
                dist[w] = new_dist
                prev[w] = v
                heapq.heappush(pq, (new_dist, w))

    return dist, prev


def reconstruct_path(prev, start, end):
    """根据 prev 表恢复最短路径"""
    path = []
    v = end
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()
    return path if path[0] == start else []


# ============================================================
# 6. 拓扑排序
# ============================================================
from collections import deque


def topological_sort_kahn(graph: DiGraph):
    """
    Kahn 算法（BFS 版）

    核心：不断删除入度为 0 的节点
      - 入度 = 0 意味着"没有前置依赖"了
      - 删掉它，更新邻居的入度
      - 重复直到所有节点处理完
    """
    # 计算每个节点的入度
    in_degree = {v: 0 for v in graph.adj}
    for v in graph.adj:
        for w, _ in graph.adj[v]:
            in_degree[w] += 1

    # 入度为 0 的入队
    q = deque([v for v, deg in in_degree.items() if deg == 0])
    result = []

    while q:
        v = q.popleft()
        result.append(v)
        for w, _ in graph.adj[v]:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                q.append(w)

    # 如果有节点没被处理 → 有环
    if len(result) != len(graph.adj):
        raise ValueError("图中存在环，无法拓扑排序")

    return result


def topological_sort_dfs(graph: DiGraph):
    """
    DFS 版拓扑排序

    核心：DFS 后序的逆序就是拓扑序
      想象一个节点，它的所有后代都先被标记完成，
      它自己最后标记 → 祖先在后序中在后
      反转后序 → 祖先在前，后代在后 ✓
    """
    UNVISITED, VISITING, VISITED = 0, 1, 2
    state = {v: UNVISITED for v in graph.adj}
    result = []

    def _dfs(v):
        if state[v] == VISITING:
            raise ValueError("图中存在环，无法拓扑排序")
        if state[v] == VISITED:
            return

        state[v] = VISITING
        for w, _ in graph.adj[v]:
            _dfs(w)
        state[v] = VISITED
        result.append(v)  # 后序

    for v in graph.adj:
        if state[v] == UNVISITED:
            _dfs(v)

    result.reverse()  # 后序的逆序
    return result


# ============================================================
# 6. 并查集 Union-Find (Disjoint Set)
# ============================================================

class UnionFind:
    """
    并查集 —— 处理"连通性"问题

    核心操作只有两个：
      - find(x): 找 x 的根（所在集合的代表）
      - union(x, y): 合并 x 和 y 所在的集合

    两个核心优化：
      - 路径压缩 (path compression): find 时直接把节点挂到根上
      - 按秩合并 (union by rank): 矮树挂到高树下
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n          # 树的高度上界
        self.count = n               # 连通分量的数量

    def find(self, x):
        """找根 + 路径压缩"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并两个集合，按秩优化"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                     # 已经在同一集合

        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx                  # 保证 rx 是较深的根
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.count -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def component_count(self):
        return self.count


# ============================================================
# 7. 最小生成树 —— Kruskal 算法
# ============================================================

def kruskal(vertices, edges):
    """
    Kruskal 算法求最小生成树 (MST)

    参数:
      vertices: list[str] — 所有顶点
      edges: list[(u, v, weight)] — 所有边（无向）

    核心思想：贪心 + 并查集
      1. 所有边按权重从小到大排序
      2. 从小到大遍历，如果边的两端不在同一集合
         → 加入 MST，合并两端
      3. 选了 n-1 条边就结束

    为什么正确？因为"最小权重且不产生环"的边一定在某个 MST 中
    """
    n = len(vertices)
    index = {v: i for i, v in enumerate(vertices)}

    uf = UnionFind(n)
    edges_sorted = sorted(edges, key=lambda e: e[2])  # 按权重排序
    mst = []          # 选中的边
    total_weight = 0

    for u, v, w in edges_sorted:
        if uf.find(index[u]) != uf.find(index[v]):
            uf.union(index[u], index[v])
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break

    return mst, total_weight


# ============================================================
# 8. Demo 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("邻接表 —— 无向图")
    print("=" * 50)
    g = GraphAdjList()
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
    for u, v in edges:
        g.add_edge(u, v)
    print(g)
    print(f"\n顶点 B 的度: {g.degree('B')}")
    print(f"DFS from A: {dfs(g, 'A')}")
    print(f"BFS from A: {bfs(g, 'A')}")

    print("\n" + "=" * 50)
    print("邻接矩阵 —— 无向图")
    print("=" * 50)
    gm = GraphAdjMatrix(5)
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]:
        gm.add_edge(u, v)
    print(gm)

    print("\n" + "=" * 50)
    print("有向无权图")
    print("=" * 50)
    dg = DiGraph()
    dg.add_edge('A', 'B')
    dg.add_edge('A', 'C')
    dg.add_edge('B', 'D')
    dg.add_edge('C', 'D')
    print(f"A 的出度: {dg.out_degree('A')}")
    print(f"D 的入度: {dg.in_degree('D')}")

    print("\n" + "=" * 50)
    print("Dijkstra 最短路径")
    print("=" * 50)
    dg2 = DiGraph()
    dg2.add_edge('A', 'B', 4)
    dg2.add_edge('A', 'C', 2)
    dg2.add_edge('B', 'C', 1)
    dg2.add_edge('B', 'D', 5)
    dg2.add_edge('C', 'D', 8)
    dg2.add_edge('C', 'E', 10)
    dg2.add_edge('D', 'E', 2)
    dg2.add_edge('D', 'F', 6)
    dg2.add_edge('E', 'F', 3)

    dist, prev = dijkstra(dg2, 'A')
    print("\n节点  | 最短距离 | 路径")
    print("-" * 35)
    for node in ['A', 'B', 'C', 'D', 'E', 'F']:
        path = reconstruct_path(prev, 'A', node)
        path_str = " → ".join(path) if path else "—"
        dist_str = f"{dist[node]:.0f}" if dist[node] != float('inf') else "∞"
        print(f"  {node}   |   {dist_str:>4s}   | {path_str}")

    print("\n" + "=" * 50)
    print("拓扑排序")
    print("=" * 50)
    # 选课依赖：A→B→D, A→C→D
    topo_g = DiGraph()
    topo_g.add_edge('数据结构', '算法')
    topo_g.add_edge('数据结构', '操作系统')
    topo_g.add_edge('算法', '机器学习')
    topo_g.add_edge('操作系统', '机器学习')
    topo_g.add_edge('机器学习', '毕业设计')

    print("Kahn 算法:", " → ".join(topological_sort_kahn(topo_g)))
    print("DFS 算法:", " → ".join(topological_sort_dfs(topo_g)))

    print("\n" + "=" * 50)
    print("并查集 Union-Find")
    print("=" * 50)
    uf = UnionFind(7)
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)]
    for u, v in edges:
        uf.union(u, v)
    print(f"0-2 是否连通: {uf.connected(0, 2)}")
    print(f"0-3 是否连通: {uf.connected(0, 3)}")
    print(f"连通分量数: {uf.component_count()}")  # 2: {0,1,2} 和 {3,4,5,6}
    uf.union(2, 3)
    print("合并 2 和 3 后:")
    print(f"0-6 是否连通: {uf.connected(0, 6)}")
    print(f"连通分量数: {uf.component_count()}")  # 1: 全部连通

    print("\n" + "=" * 50)
    print("Kruskal 最小生成树")
    print("=" * 50)
    vertices = ['A', 'B', 'C', 'D', 'E', 'F']
    k_edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6),
        ('E', 'F', 3),
    ]
    mst, total = kruskal(vertices, k_edges)
    print(f"MST 总权重: {total}")
    for u, v, w in mst:
        print(f"  {u} — {v}  ({w})")
