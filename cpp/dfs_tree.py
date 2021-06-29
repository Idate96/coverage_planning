from collections import defaultdict
from typing import Tuple, List


class Graph(object):
    def __init__(self, adj_list=None) -> None:
        super().__init__()
        self.adj = defaultdict(list)
        if adj_list:
            self.adj = adj_list

    def add_edge(self, u, v, directed=True):
        self.adj[u].append(v)
        if not directed:
            self.adj[v].append(u)

    def add_edges(self, edges: List[Tuple[int, int]], directed=True):
        for edge in edges:
            self.add_edge(edge[0], edge[1], directed=directed)

    def DFS(self, root) -> List[int]:
        """Depth First Search traversal of the graph

        Args:
            root ([type]): node from which to start the search

        Returns:
            List[int]: list of visited nodes
        """
        visited = list([root])

        def dfs_(node, visited):
            for neighboor in self.adj[node]:
                if not neighboor in visited:
                    visited.append(neighboor)
                    dfs_(neighboor, visited)

        dfs_(root, visited)
        return visited

    def DFS_tree(self, root) -> List[Tuple[int, int]]:
        """Traverses the graph in a DFS fashion to generate a spanning tree
        with a minimal amount of branching

        Args:
            root ([type]): node from which to start the search

        Returns:
            List[Tuple[int, int]]: list of edge making up a spanning tree of the graph
        """
        visited = list([root])
        tree = list()

        def dfs_(node, visited):
            for neighboor in self.adj[node]:
                if not neighboor in visited:
                    visited.append(neighboor)
                    tree.append((node, neighboor))
                    dfs_(neighboor, visited)

        dfs_(root, visited)
        return tree

    def find_height(self, root) -> Tuple[int, int]:
        """Use DFS to find the height of the tree.

        Args:
            root (int): random initial node from which to run DFS

        Returns:
            furthest_node (int): furthest away node from the root
            max_tree_height(int): diameter of the tree (longest path length)
        """
        max_tree_height = 0
        furthest_node = root
        visited = list([root])

        def dfs_(node, visited, tree_height):
            nonlocal max_tree_height, furthest_node

            for neighboor in self.adj[node]:
                if not neighboor in visited:
                    visited.append(neighboor)
                    if tree_height > max_tree_height:
                        max_tree_height = tree_height
                        furthest_node = neighboor
                    dfs_(neighboor, visited, tree_height + 1)
            tree_height -= 1

        dfs_(root, visited, 1)
        return max_tree_height, furthest_node

    def find_diameter(self, node) -> Tuple[int, int]:
        """Returns the diameter of the tree

        Args:
            node (int): random node
        Returns:
            diameter (int): maximum path length in the tree
            furthest_node (int): node furtherest away from the root
        """
        # first find furthest node
        _, new_root = self.find_height(node)
        # assume it's an undirected tree
        diameter, furthest_node = self.find_height(new_root)
        return diameter, new_root, furthest_node

    def post_order_traversal(self, node) -> List[int]:
        """Post order traversal of the graph. A node is marked visited once its children have been visited


        Args:
            node (int): node from which to start the traversal 

        Returns:
            List[int]: visiting order of the nodes 
        """
        visited = list()

        def dfs_(node, visited):
            for neighboor in self.adj[node]:
                dfs_(neighboor, visited)
            visited.append(node)
        
        dfs_(node, visited)
        return visited



