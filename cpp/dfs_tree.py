from collections import defaultdict

class Graph(object):
    def __init__(self, adj_list=None) -> None:
        super().__init__()
        self.adj = defaultdict(list)
        if adj_list:
            self.adj = adj_list

    def add_edge(self, u, v):
        self.adj[u].append(v)


    def DFS(self, root):
        visited = list(root)

        def dfs_(node, visited):
            for neighboor in self.adj[node]:
                if not node in visited:
                    visited.append(node)
                    dfs_(neighboor, visited)
        
        dfs_(root, visited)
        return visited

