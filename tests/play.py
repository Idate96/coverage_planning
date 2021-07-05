import matplotlib.pyplot as plt
from cpp.dfs_tree import Graph
import matplotlib.pyplot as plt

path = []
for i in range(10):
    path.append((i, 1 / 2 * i ** 2))

x, y = zip(*path)

print(x, y)


def calculate_days_between_dates(begin_data, end_date):
    """
    Calculate number of days between two dates
    :param begin_data: date, begin date
    :param end_date: date, end date
    :return: int, number of days
    """
    d1 = begin_data
    d2 = end_date
    return (d2 - d1).days


import datetime

date_1 = datetime.date(2018, 5, 1)
date_2 = datetime.date(2018, 5, 15)

print(calculate_days_between_dates(date_1, date_2))


def find_biggest_common_substring(string_1, string_2):
    """
    Find the biggest common substring between two strings
    :param string_1: str, first string
    :param string_2: str, second string
    :return: str, biggest common substring
    """
    max_len = 0
    for i in range(len(string_1)):
        for j in range(len(string_2)):
            if string_1[i : i + len(string_2)] == string_2[j : j + len(string_2)]:
                if len(string_1[i : i + len(string_2)]) > max_len:
                    max_len = len(string_1[i : i + len(string_2)])
                    res = string_1[i : i + len(string_2)]
    return res


print(find_biggest_common_substring("hello", "che bello"))


def depth_first_search(graph: Graph):
    """
    Depth first search
    :param graph: Graph, graph
    :return: list, path
    """
    global path
    path = []
    stack = []
    stack.append(graph.get_vertex(0))
    while stack:
        actual_vertex = stack.pop()
        if (
            actual_vertex.get_id()
            == graph.get_vertex(graph.get_num_vertices() - 1).get_id()
        ):
            path.append(actual_vertex.get_id())
            return path
        for neighbor in actual_vertex.get_connections():
            if neighbor not in stack:
                stack.append(neighbor)
    return path


def find_closest_point(point, points):
    """
    Find the closest point to the given point
    :param point: tuple, point
    :param points: list, points
    :return: tuple, closest point
    """
    min_dist = 1000000
    closest_point = None
    for p in points:
        dist = ((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_point = p
    return closest_point


def find_prime_numbers_in_range(range: int):
    """
    Find prime numbers in range
    :param range: int, range
    :return: list, prime numbers
    """
    prime_numbers = []
    for i in range(range):
        if is_prime(i):
            prime_numbers.append(i)
    return prime_numbers


def is_prime(num: int):
    """
    Check if number is prime
    :param num: int, number
    :return: bool, True if prime, False otherwise
    """
    if num == 1:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True


def find_shortest_path(node_start, node_end):
    """
    Find shortest path between two nodes
    :param node_start: Node, start node
    :param node_end: Node, end node
    :return: list, shortest path
    """
    global path
    path = []
    stack = []
    stack.append(node_start)
    while stack:
        actual_node = stack.pop()
        if actual_node.get_id() == node_end.get_id():
            path.append(actual_node.get_id())
            return path
        for neighbor in actual_node.get_connections():
            if neighbor not in stack:
                stack.append(neighbor)
    return path


def is_prime(number: int) -> bool:
    """
    Check if number is prime
    :param number: int, number
    :return: bool, True if prime, False otherwise
    """
    if number == 1:
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True


for i in range(1, 20):
    print(i, is_prime(i))


def plot_shortest_path(node_start, node_end):
    """
    Plot shortest path between two nodes
    :param node_start: Node, start node
    :param node_end: Node, end node
    :return: None
    """
    path = find_shortest_path(node_start, node_end)
    x, y = zip(*path)
    plt.plot(x, y)
    plt.show()


def plot_surface(z, x, y):
    """Plot Surface

    Args:
        z ([type]): House prices
        x ([type]): Meter squares
        y ([type]): Number of rooms
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("Meter squares")
    ax.set_ylabel("Number of rooms")
    ax.set_zlabel("House prices")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def longest_common_substring(string_1, string_2):
    """Find the longest common substring with dynamic programming

    Args:
        string_1 ([type]): first string
        string_2 ([type]): second string
    """
    m = len(string_1)
    n = len(string_2)
    max_len = 0
    for i in range(m):
        for j in range(n):
            if string_1[i] == string_2[j]:
                if i > 0 and j > 0:
                    if string_1[i - 1] == string_2[j - 1]:
                        max_len = max(
                            max_len,
                            2
                            + longest_common_substring(
                                string_1[i - 1 : i + 1], string_2[j - 1 : j + 1]
                            ),
                        )
                    else:
                        max_len = max(
                            max_len,
                            1
                            + longest_common_substring(
                                string_1[i : i + 1], string_2[j : j + 1]
                            ),
                        )
                else:
                    max_len = max(max_len, 1)
    return max_len
