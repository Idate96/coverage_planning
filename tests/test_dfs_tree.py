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


def test_graph():
    graph = create_graph()
    assert graph.adj[0] == [1, 2]


def test_dfs():
    graph = create_graph()
    assert graph.DFS(0) == [0, 1, 3, 6, 4, 2, 5]


if __name__ == "__main__":
    test_dfs()
