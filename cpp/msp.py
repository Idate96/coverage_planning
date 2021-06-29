class Node :

    def __init__(self, arg_id) :
        self._id = arg_id

class Graph :

    def __init__(self, arg_source, arg_adj_list) :
        self.source = arg_source
        self.adjlist = arg_adj_list

    def PrimsMST(self):

        # Priority queue is implemented as a dictionary with
        # key as an object of 'Node' class and value as the cost of 
        # reaching the node from the source.
        # Since the priority queue can have multiple entries for the
        # same adjacent node but a different cost, we have to use objects as
        # keys so that they can be stored in a dictionary. 
        # [As dictionary can't have duplicate keys so objectify the key]

        # The distance of source node from itself is 0. Add source node as the first node
        # in the priority queue
        priority_queue = { Node(self.source) : 0 }
        added = [False] * len(self.adjlist)
        min_span_tree_cost = 0
        
        while priority_queue :
            # Choose the adjacent node with the least edge cost
            node = min (priority_queue, key=priority_queue.get)
            cost = priority_queue[node]

            # Remove the node from the priority queue
            del priority_queue[node]

            if added[node._id] == False :
                min_span_tree_cost += cost
                added[node._id] = True
                print("Added Node : " + str(node._id) + ", cost now : "+str(min_span_tree_cost))

                for item in self.adjlist[node._id] :
                    adjnode = item[0]
                    adjcost = item[1]
                    if added[adjnode] == False :
                        # is it not missing the starting node ?
                        # no overwriting ? 
                        priority_queue[Node(adjnode)] = adjcost

        return min_span_tree_cost