
def find_centroid_poligon(vertexes: List[Tuple[int, int]]):
    """Find the centroid"""
    x_sum = 0
    y_sum = 0
    for x, y in vertexes:
        x_sum += x
        y_sum += y
    x_sum /= len(vertexes)
    y_sum /= len(vertexes)
    return x_sum, y_sum


def find_common_substring(string_1, string_2):
    """Find the common substring"""
    common_substring = ""
    for i in range(len(string_1)):
        if string_1[i] == string_2[i]:
            common_substring += string_1[i]
    return common_substring

def find_common_substring(string_1, string_2):
    """Find the common substring with dynamics programming"""
    common_substring = ""
    for i in range(len(string_1)):
        if string_1[i] == string_2[i]:
            common_substring += string_1[i]
    return common_substring

class Graph:
    """Graph class"""
    def __init__(self, vertexes: List[Tuple[int, int]], edges: List[Tuple[int, int]]):
        self.vertexes = vertexes
        self.edges = edges
        self.adj_list = self.__create_adj_list()

    def __create_adj_list(self):
        """Create adjacency list"""
        adj_list = {}
        for edge in self.edges:
            if edge[0] in adj_list:
                adj_list[edge[0]].append(edge[1])
            else:
                adj_list[edge[0]] = [edge[1]]
            if edge[1] in adj_list:
                adj_list[edge[1]].append(edge[0])
            else:
                adj_list[edge[1]] = [edge[0]]
        return adj_list

    def __str__(self):
        """String representation"""
        return str(self.adj_list)

    def __repr__(self):
        """Representation"""
        return str(self.adj_list)

    def __len__(self):
        """Length"""
        return len(self.adj_list)

    def __contains__(self, item):
        """Contains"""
        return item in self.adj_list

    def __getitem__(self, item):
        """Get item"""
        return self.adj_list[item]

    def __iter__(self):
        """Iterate"""
        return iter(self.adj_list)

    def __delitem__(self, key):
        """Delete item"""
        del self.adj_list[key]

    def __setitem__(self, key, value):
        """Set item"""
        self.adj_list[key] = value

    def __reversed__(self):
        """Reverse"""
        return reversed(self.adj_list)

    def __add__(self, other):
        """Add"""
        return self.adj_list + other

    def __radd__(self, other)
    