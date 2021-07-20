from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np
from collections import defaultdict, namedtuple

Arc = namedtuple("Arc", ("tail", "weight", "head"))


class Graph(object):
    def __init__(
        self,
        adj_list: Dict[int, List[int]] = None,
        weights: Dict[int, Dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.adj = defaultdict(list)
        self.weights = defaultdict(lambda: defaultdict(int))

        if adj_list:
            self.adj = adj_list
            if weights:
                self.weights = weights

    def add_edge(self, u, v, directed=True, weight=None):
        # check if v is in adj
        if v not in self.adj:
            self.adj[v] = list()

        self.adj[u].append(v)
        if not directed:
            self.adj[v].append(u)

        if weight:
            self.weights[u][v] = weight

    def add_edges(
        self, edges: List[Tuple[int, int]], directed=True, weights: List[int] = None
    ) -> None:
        if weights:
            for i, edge in enumerate(edges):
                self.add_edge(edge[0], edge[1], directed=directed, weight=weights[i])
        else:
            for i, edge in enumerate(edges):
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
        # TODO: works correctly for trees but not for directed graphs

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

    def post_order_traversal_sorted(self, node, original_graph) -> List[int]:
        """Post order traversal of the graph.
        While doing the DFS the neighboors are sorted using the distance from the last visited node.
        The sorting is done by running djikstra's algorithm on the original graph.
        """

        visited = list()

        def dfs_(node, visited):
            if visited:
                last_visited = visited[-1]
                # sort nodes at equal heights (distance from last visited node)
                neighboors = sorted(
                    self.adj[node],
                    key=lambda x: original_graph.djikstra(last_visited, x),
                )
            else:
                neighboors = self.adj[node]

            for neighboor in neighboors:
                dfs_(neighboor, visited)
            visited.append(node)

        dfs_(node, visited)
        return visited

    def graph_to_arcs(self) -> List[Arc]:
        """Represent the graph as list of directed edges (arcs)

        Returns:
            List[Arc]: list of directed edges with weights
        """
        arcs = list()
        for node in self.adj:
            for neighboor in self.adj[node]:
                # edges are unweighted
                arcs.append(Arc(node, 1, neighboor))
        return arcs

    def graph_to_arcs_with_dfs(self, source) -> List[Arc]:
        """Represent the graph as a list of directed edges (arcs) by
        traversing the tree with depth first search (dfs).

        Returns:
            List[Arc]: list of directed edges with weights
        """
        visited = list()

        def dfs_(node, visited_nodes, visited_edges):
            for neighboor in self.adj[node]:
                if not neighboor in visited_nodes:
                    visited_nodes.append(neighboor)
                    arc = Arc(node, 1, neighboor)
                    if arc not in visited_edges:
                        visited_edges.append(arc)
                        dfs_(neighboor, visited_nodes, visited_edges)

            for neighboor in self.adj[node]:
                if neighboor in visited_nodes:
                    arc = Arc(node, 1, neighboor)
                    if arc not in visited_edges:
                        visited_edges.append(arc)
                        dfs_(neighboor, visited_nodes, visited_edges)

        arcs = list()
        # graph has at least one node
        if self.adj:
            dfs_(source, [source], arcs)

        # add nodes that are not discovred with dfs
        for node in self.adj:
            if node not in visited:
                if not self.node_has_only_undirected_edges(node):
                    visited.append(node)
                    dfs_(node, visited, arcs)

        return arcs

    def arcs_to_graph(self, arcs: List[Arc]):
        """Generate a graph

        Args:
            arcs (List[Arc]): list of directed edges
        """
        self.adj = defaultdict(list)
        for arc in arcs:
            self.add_edge(arc.tail, arc.head, directed=True)

    def node_has_only_undirected_edges(self, node) -> bool:
        """Check if a node has only undirected edges

        Args:
            node (int): node

        Returns:
            bool: True if the node has only undirected edges
        """
        for neighboor in self.adj[node]:
            if node not in self.adj[neighboor]:
                return False
        return True

    def reverse_graph_edges(self):
        """Reverse the direction of all the edges in the graph"""
        tmp_adj = self.adj.copy()
        self.adj = defaultdict(list)
        for node in tmp_adj:
            for neighboor in tmp_adj[node]:
                self.add_edge(neighboor, node, directed=True)

    def djikstra(self, source: int, target: int) -> int:
        """Find the shortest path from source to target using Dijkstra's algorithm"""
        # init
        visited = list()
        dist = dict()
        prev = dict()
        for node in self.adj:
            dist[node] = float("inf")
            prev[node] = None
        dist[source] = 0
        prev[source] = None
        # run
        while len(visited) < len(self.adj):
            # get node with minimum distance that is not visited
            node = min(
                dist, key=lambda x: dist[x] if x not in visited else float("inf")
            )
            visited.append(node)
            # update distances
            for neighboor in self.adj[node]:
                if neighboor in visited:
                    continue
                alt = dist[node] + self.weights[node][neighboor]
                if alt < dist[neighboor]:
                    dist[neighboor] = alt
                    prev[neighboor] = node

        # find path

        path = list()
        curr = target
        while curr != source:
            path.append(curr)
            curr = prev[curr]
        path.append(curr)
        return dist[target], path


def undirected_graph(graph: Graph, copy_weights=False) -> Graph:
    """Generate an undirected graph from a directed graph

    Args:
        graph (Graph): directed graph

    Returns:
        graph_undirected: undirected graph
    """
    undirected_graph = defaultdict(list)
    undirected_graph_weights = defaultdict(lambda: defaultdict(int))
    for node in graph.adj:
        for neighboor in graph.adj[node]:
            undirected_graph[node].append(neighboor)
            undirected_graph[neighboor].append(node)

    if copy_weights:
        for node in graph.adj:
            for neighboor in graph.adj[node]:
                undirected_graph_weights[node][neighboor] = graph.weights[node][
                    neighboor
                ]
                undirected_graph_weights[neighboor][node] = graph.weights[node][
                    neighboor
                ]

    return Graph(undirected_graph, weights=undirected_graph_weights)


def connected_cells(adj_matrix: np.ndarray, cell_1, cell_2) -> bool:
    """Check if two cells are connected

    Args:
        adj_matrix (np.ndarray): adjacency matrix
        cell_1 (int): first cell
        cell_2 (int): second cell

    Returns:
        bool: True if the cells are connected
    """
    visited = list()
    queue = list()
    queue.append(cell_1)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            for neighboor in adj_matrix[node]:
                if neighboor != cell_1:
                    queue.append(neighboor)
    return cell_2 in visited


def adj_matrix_to_dict(adj_matrix: np.array) -> Dict[int, List[int]]:
    """Convert adjacency matrix to a dict of lists.
    This is used since the Graph class constructor only takes a list of edges.

    Args:
        adj_matrix (np.array): graph as adjacency matrix

    Returns:
        graph_dict (Dict[int, List[int]]): graph as adjacency list
    """
    adj_matrix = adj_matrix.astype(np.bool)
    graph_dict = defaultdict(list)
    n, _ = np.shape(adj_matrix)

    for node_id in range(n):
        graph_dict[node_id] = []
        for neighboor_id in range(n):
            if adj_matrix[node_id, neighboor_id]:
                graph_dict[node_id].append(neighboor_id)

    return graph_dict


# find a minimum spanning tree of weighted graph
# from https://stackoverflow.com/questions/23988236/chu-liu-edmonds-algorithm-for-minimum-spanning-tree-on-directed-graphs/38757262


def min_spanning_arborescence(arcs, sink):
    """Find a minimum spanning arborescence of directed graph.
    The algorithm should run in O(EV) time.

    Args:
        arcs ([type]): [description]
        sink ([type]): [description]

    Returns:
        [type]: [description]
    """
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if (
                tail_rep not in min_arc_by_tail_rep
                or min_arc_by_tail_rep[tail_rep].weight > arc.weight
            ):
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {
            node: cycle_rep if node_rep in cycle_rep_set else node_rep
            for node, node_rep in quotient_map.items()
        }


def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node) :]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail


if __name__ == "__main__":
    list_arcs = [
        Arc(0, 1, 1),
        Arc(1, 1, 0),
        Arc(1, 1, 5),
        Arc(5, 1, 1),
        Arc(5, 1, 3),
        Arc(3, 1, 5),
        Arc(4, 1, 5),
        Arc(2, 1, 0),
        Arc(3, 1, 0),
        Arc(3, 1, 0),
    ]

    print(min_spanning_arborescence(list_arcs, 0))
