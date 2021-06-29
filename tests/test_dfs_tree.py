from cpp.dfs_tree import Graph


def create_graph():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 6)
    return graph


def create_graph_2():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 6)
    graph.add_edge(6, 4)
    graph.add_edge(4, 2)
    return graph


def create_graph_undirected():
    graph = Graph()
    graph.add_edge(0, 1, directed=False)
    graph.add_edge(0, 2, directed=False)
    graph.add_edge(1, 3, directed=False)
    graph.add_edge(1, 4, directed=False)
    graph.add_edge(2, 5, directed=False)
    graph.add_edge(3, 6, directed=False)
    return graph


def create_graph_undirected_2():
    graph = Graph()
    graph.add_edge(0, 1, directed=False)
    graph.add_edge(0, 2, directed=False)
    graph.add_edge(1, 3, directed=False)
    graph.add_edge(1, 4, directed=False)
    graph.add_edge(2, 5, directed=False)
    graph.add_edge(3, 6, directed=False)
    graph.add_edge(6, 4, directed=False)
    graph.add_edge(4, 2, directed=False)
    return graph


def test_graph():
    graph = create_graph()
    assert graph.adj[0] == [1, 2]


def test_dfs():
    graph = create_graph()
    assert graph.DFS(0) == [0, 1, 3, 6, 4, 2, 5]


def test_dfs_undirected():
    graph = create_graph_undirected()
    assert graph.DFS(0) == [0, 1, 3, 6, 4, 2, 5]


def test_dfs_tree():
    graph = create_graph()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (1, 4), (0, 2), (2, 5)]


def test_dfs_tree_undirected():
    graph = create_graph_undirected()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (1, 4), (0, 2), (2, 5)]


def test_dfs_graph():
    graph = create_graph_2()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (6, 4), (4, 2), (2, 5)]


def test_dfs_graph_undirected():
    graph = create_graph_undirected_2()
    assert graph.DFS_tree(0) == [(0, 1), (1, 3), (3, 6), (6, 4), (4, 2), (2, 5)]


def test_find_height():
    graph = create_graph_2()
    tree_edges = graph.DFS_tree(0)
    graph_tree = Graph()
    graph_tree.add_edges(tree_edges)
    assert graph_tree.find_height(0) == (6, 5)


def test_find_diameter():
    tree = create_graph_undirected()
    assert tree.find_diameter(0) == (5, 6, 5)


if __name__ == "__main__":
    test_find_diameter()
