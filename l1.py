import random
from heapq import heappush, heappop
from collections import deque
import sys


class DirectedGraph:
    """
    A class for an object corresponding to a directed graph.
    """

    def __init__(self, vertices, edges):
        """
        Creates an instance of a directed graph with the corresponding parameters.
        :param vertices: nr. of vertices of the graph
        :param edges: nr. of edges
        The graph will consist of 2 dictionaries ( corresponding to the in/ out degrees, a dictionary which will store
        the associated costs; the dictionaries will be initialised with empty lists, then the corresponding starting and
        ending point will be appended
        """
        self.__in = {}
        self.__out = {}
        self.__costs = {}
        self.negative_costs = {}

        for v in vertices:
            self.__in[v] = []
            self.__out[v] = []
        print(2)
        for e in edges:
            self.__out[e[0]].append(e[1])
            self.__in[e[1]].append(e[0])
            if e[2] is not None:
                self.__costs[(e[0], e[1])] = e[2]
                self.negative_costs[(e[0], e[1])] = e[2] * (-1)

    def get_vertex_count(self):
        """
        :return: nr. of vertices in the graph ( self.__in == self.__out )
        """
        return len(self.__in)

    def get_edges_count(self):
        """
        :return: nr. of edges in the graph
        """
        return sum([len(x) for x in self.__out.values()])

    def get_vertices(self):
        """
        :return: a new list which contains all the vertices
        """
        return list(self.__in.keys())

    def is_edge(self, start, end):
        """
        Checks if a given edge exists in the current graph or not.
        :param start: this vertex is considered outbound
        :param end: this vertex is considered inbound
        :return: true or false
        """
        return start in self.__out and end in self.__out[start]

    def get_out_degree(self, vertex):
        """
        :param vertex: an integer
        :return: out degree of the vertex
        """
        return len(self.__out[vertex])

    def get_in(self, vertex):
        lst = []
        for v in self.get_vertices():
            if self.is_edge(v, vertex):
                lst.append(v)
        return lst

    def get_out(self, vertex):
        lst = []
        for v in self.get_vertices():
            if self.is_edge(vertex, v):
                lst.append(v)
        return lst

    def get_in_degree(self, vertex):
        """
        :param vertex: an integer representing the vertex
        :return: in degree of the vertex
        """
        return len(self.__in[vertex])

    def get_out_vertices(self, start):
        """
        :param start: the vertex to which the edges go
        :return: a list of vertices
        """
        return list(self.__out[start])

    def get_in_vertices(self, end):
        """
        :param end: the vertex from which the edges start
        :return: a list of vertices
        """
        return list(self.__in[end])

    def get_negative_cost(self, start, end):
        return self.negative_costs[(start, end)]

    def get_cost(self, start, end):
        """
        :param start: an integer representing the vertex
        :param end: an integer representing the vertex
        :return: the cost of the edge defined by the starting and ending vertices
        """
        return self.__costs[(start, end)]

    def set_cost(self, start, end, cost):
        """
        Sets a new cost to a given edge.
        :param start: an integer representing the vertex
        :param end: an integer representing the vertex
        :param cost: an integer representing the cost
        :return: the modified entity
        """
        self.__costs[(start, end)] = cost
        return self

    def add_vertex(self, vertex):
        """
        Adds a new vertex to the graph.
        :param vertex: an integer
        """
        self.__in[vertex] = []
        self.__out[vertex] = []

    def add_edge(self, start, end, cost):
        """
        Adds a new edge to the graph. ( defined by vertices start and end, with an associated cost )
        """
        if start not in self.__in:
            self.add_vertex(start)
        if end not in self.__in:
            self.add_vertex(end)
        self.__out[start].append(end)
        self.__in[end].append(start)
        self.__costs[(start, end)] = cost

    def remove_edge(self, start, end):
        """
        Removes a given edge.
        :param start: integer ( a vertex )
        :param end: integer ( a vertex )
        """
        self.__out[start].remove(end)
        self.__in[end].remove(start)
        self.__costs.pop((start, end))

    def remove_vertex(self, vertex):
        """
        Removes an existing vertex from the graph.
        :param vertex: integer value
        """
        for v in self.get_out_vertices(vertex):
            self.remove_edge(vertex, v)
        for v in self.get_in_vertices(vertex):
            if self.is_edge(v, vertex):
                self.remove_edge(v, vertex)
        self.__in.pop(vertex)
        self.__out.pop(vertex)

    def create_copy(self):
        """
        Creates a copy of the already existing graph.
        :return: the new graph
        """
        vertices = self.get_vertices()
        g = DirectedGraph(vertices, [])
        for v1 in vertices:
            for v2 in self.get_out_vertices(v1):
                g.add_edge(v1, v2, self.get_cost(v1, v2) * (-1))
        return g

    def change_sign(self):
        for cost in self.__costs:
            cost *= (-1)


def bfs_backward(graph, end):
    """
    Backward Breadth-first search of the graph( the parameter ), starting with the target vertex.
    :param graph: a given graph
    :param end: a vertex from which the bfs is conducted
    :return: the distances from the ending vertex ( dictionary ),  the previous vertex on the path ( dictionary )
    """
    source = end
    queue = [source]
    index = 0
    distances = {}
    before = {source: None}
    visited = {source}
    distances[source] = 0
    while index < len(queue):
        current = queue[index]
        for x in graph.get_in(current):
            if x not in visited:
                queue.append(x)
                visited.add(x)
                distances[x] = distances[current] + 1
                before[x] = current
        index += 1
    # print(distances)
    return distances, before


def df_traversal(graph, vertex, visited, result):
    if visited[vertex] is False:
        neighbours = graph.get_out(vertex)
        visited[vertex] = True
        for neighbour_vertex in neighbours:
            df_traversal(graph, neighbour_vertex, visited, result)
        result.append(vertex)


def tarjan_toposort(graph):
    result = []
    visited = {}
    graph_length = graph.get_vertex_count()
    is_dag = True
    for vertex in graph.get_vertices():
        visited[vertex] = False

    for vertex in graph.get_vertices():
        if visited[vertex] is False:
            df_traversal(graph, vertex, visited, result)

    if not result:
        print("The graph is not a DAG.")
        is_dag = False
    else:
        print("The graph is a DAG.\nTopological sort: ")

    print(result)
    return is_dag, result


def highest_cost_path(graph):
    is_dag, toposort_path = tarjan_toposort(graph)
    length = len(toposort_path) - 1
    source = toposort_path[0]
    destination = toposort_path[1]

    negative_copy = graph.create_copy()
    maximum_cost_path = dijkstra_lowest_cost_path(negative_copy, source, destination)

    while maximum_cost_path is None:
        length -= 1
        destination = toposort_path[length]
        maximum_cost_path = dijkstra_lowest_cost_path(graph, source, destination)

    return maximum_cost_path


def shortest_path_2(g, source, target):
    """
    Determines the optimal ( shortest ) path from source to target vertex. ( minimum number of edges )
    :param g: given graph
    :param source: start vertex
    :param target: end vertex
    :return: a list representing the path ; if no path exists, returns None
    """
    distances, _ = bfs_backward(g, target)
    if source not in distances:
        return None
    path = []
    current_vertex = source
    while current_vertex != target:
        path.append(current_vertex)
        for neighbour in g.get_out(current_vertex):
            if neighbour in distances and distances[neighbour] == distances[current_vertex] - 1:
                current_vertex = neighbour
                break
    path.append(target)
    return path


def dijkstra(graph, start_node, end_node=None):
    distances = {}
    previous_nodes = {}
    distances[start_node] = 0
    previous_nodes[start_node] = None
    priority_queue = []
    heappush(priority_queue, (0, start_node))

    while len(priority_queue) > 0:
        cost, node = heappop(priority_queue)
        if node == end_node:
            break
        if cost != distances[node]:
            continue
        for next_node in graph.get_out(node):
            edge_cost = graph.get_cost(node, next_node)
            if next_node not in distances or distances[next_node] < cost + edge_cost:
                # if next_node not in distances or distances[next_node] > cost + edge_cost:
                distances[next_node] = cost + edge_cost
                previous_nodes[next_node] = node
                heappush(priority_queue, (distances[next_node], next_node))
    result = (distances, previous_nodes)
    return result


def dijkstra_lowest_cost_path(graph, start_vertex, destination_vertex):
    """
    Using Dijkstra's algorithm, computes the minimum cost path in graph g between 2 vertices.
    Returns:  the list of vertices starting with start_vertex and ending with destination_vertex, also the specific cost.
    If the start and ending are the same, the function will return just the start vertex.
    If there is no path between them, it returns None.
    """
    dist, _ = dijkstra(graph, start_vertex, destination_vertex)
    if destination_vertex not in dist:
        return None

    path = []
    current_vertex = destination_vertex
    while current_vertex != start_vertex:
        path.append(current_vertex)
        for neighbour in graph.get_in(current_vertex):
            if neighbour in dist and (
                    dist[neighbour] == dist[current_vertex] - graph.get_cost(neighbour, current_vertex)):
                current_vertex = neighbour
                break
    path.append(start_vertex)
    path.reverse()
    return path, dist[destination_vertex] * (-1)
    # return path, dist[destination_vertex]


def read_from_file(path):
    """
    The function reads a graph from a file.
    :param path: the name of the file
    :return: the graph
    Then, the file will be closed.
    """
    f = open(path, 'r')
    graph = DirectedGraph([], [])
    header = f.readline()
    [vertex_count, edge_count] = [int(x) for x in header.split(' ')]
    for v in range(vertex_count):
        graph.add_vertex(v)
    for _ in range(edge_count):
        [start, end, cost] = [int(x) for x in f.readline().split(' ')]
        graph.add_edge(start, end, cost)
    f.close()
    return graph


def write_to_file(path, graph: DirectedGraph):
    """
    The function writes a graph to a file.
    If the file doesn't already exists, it will create a new one and write in it.
    Then, the file will be closed.
    :param path: file's name
    :param graph: th graph to be written
    """
    f = open(path, 'w')
    f.write(f"{graph.get_vertex_count()} {graph.get_edges_count()}\n")
    vertices = graph.get_vertices()
    for v1 in vertices:
        for v2 in graph.get_out_vertices(v1):
            f.write(f"{v1} "
                    f"{v2} "
                    f"{graph.get_cost(v1, v2)}\n")
    f.close()


def random_graph(vertex_count, edge_count):
    """
    Creates a random graph with specific parameters.
    ( the random part consists of assigning random edges until the specified number is done )
    :param vertex_count: the number of vertices
    :param edge_count: the number of edges
    :return: the randomly created new graph
    """
    graph = DirectedGraph([], [])
    for v in range(vertex_count):
        graph.add_vertex(v)
    for _ in range(edge_count):
        v1 = random.randrange(vertex_count)
        v2 = random.randrange(vertex_count)
        cost = random.randrange(1000)
        if not graph.is_edge(v1, v2):
            graph.add_edge(v1, v2, cost)
    return graph


def dijkstra_negative(graph, start_node, end_node=None):
    distances = {}
    previous_nodes = {}
    distances[start_node] = 0
    previous_nodes[start_node] = None
    priority_queue = []
    heappush(priority_queue, (0, start_node))

    while len(priority_queue) > 0:
        cost, node = heappop(priority_queue)
        if node == end_node:
            break
        if cost != distances[node]:
            continue
        for next_node in graph.get_out(node):
            edge_cost = graph.get_negative_cost(node, next_node)
            if next_node not in distances or distances[next_node] < cost + edge_cost:
                distances[next_node] = cost + edge_cost
                previous_nodes[next_node] = node
                heappush(priority_queue, (distances[next_node], next_node))
    result = (distances, previous_nodes)
    return result


def dijkstra_lowest_cost_path_negative(graph, start_vertex, destination_vertex):
    """
    Using Dijkstra's algorithm, computes the minimum cost path in graph g between 2 vertices.
    Returns:  the list of vertices starting with start_vertex and ending with destination_vertex, also the specific cost.
    If the start and ending are the same, the function will return just the start vertex.
    If there is no path between them, it returns None.
    """
    dist, _ = dijkstra_negative(graph, start_vertex, destination_vertex)
    if destination_vertex not in dist:
        return None

    path = []
    current_vertex = destination_vertex
    while current_vertex != start_vertex:
        path.append(current_vertex)
        for neighbour in graph.get_in(current_vertex):
            if neighbour in dist and (
                    dist[neighbour] == dist[current_vertex] - graph.get_negative_cost(neighbour, current_vertex)):
                current_vertex = neighbour
                break
    path.append(start_vertex)
    path.reverse()
    print(999)
    print(path)
    print(dist[destination_vertex])
    return path, dist[destination_vertex]


def run():
    """
    The menu function (ui) which will also run the whole program, until '0' is pressed in order to close the program.
    """
    print("A graph with no vertices and edges has been created.")
    graph = DirectedGraph([], [])
    done = False

    while not done:
        print(
            "~~~~~~~~~~MENU~~~~~~~~~~\n"
            "1.  Add vertex\n"
            "2.  Add edge\n"
            "3.  Get in/out degree of a vertex\n"
            "4.  Display all in/outbound edges of a vertex\n"
            "5.  Check if an edge exists\n"
            "6.  Get cost for edge\n"
            "7.  Set cost for edge\n"
            "8.  Remove vertex\n"
            "9.  Remove edge\n"
            "10. Read from file\n"
            "11. Write to file\n"
            "12. Generate a random graph\n"
            "13. Given two vertices, finds a lowest length path between them, by using a backward breadth-first search "
            "from the ending vertex.\n"
            "14. To compute the lowest cost walk between 2 vertices using Dijkstra's algorithm, press 14.\n"
            "15. Verify if the graph is a DAG(directed acyclic graph); if the graph is a DAG, the maximum cost path "
            "will be displayed\n"
            "0.  Exit"
        )

        option = input("Option: ")

        if option == "1":
            v = input("Vertex name: ")
            graph.add_vertex(int(v))
        elif option == "2":
            [s, e, c] = input("Edge ends and cost, separated by spaces: ").split(' ')
            graph.add_edge(int(s), int(e), int(c))
        elif option == "3":
            v = int(input("Vertex: "))
            print(f"In degree: {graph.get_in_degree(v)}\n"
                  f"Out degree: {graph.get_out_degree(v)}")
        elif option == "4":
            v = int(input("Vertex: "))
            print(f"Outbound: {graph.get_out_vertices(v)}\n"
                  f"Inbound: {graph.get_in_vertices(v)}")
        elif option == "5":
            [s, e] = input("Edge ends, separated by space: ").split(' ')
            print(f"Is edge: {graph.is_edge(int(s), int(e))}")
        elif option == "6":
            [s, e] = input("Edge ends, separated by space: ").split(' ')
            print(f"Cost: {graph.get_cost(int(s), int(e))}")
        elif option == "7":
            [s, e, c] = input("Edge ends and new cost, separated by spaces: ").split(' ')
            graph.set_cost(int(s), int(e), int(c))
        elif option == "8":
            v = int(input("Vertex: "))
            graph.remove_vertex(int(v))
        elif option == "9":
            [s, e] = input("Edge ends, separated by space: ").split(' ')
            graph.remove_edge(int(s), int(e))
        elif option == "10":
            file = input("File name: ")
            graph = read_from_file(file)
        elif option == "11":
            file = input("File name: ")
            write_to_file(file, graph)
        elif option == "12":
            [v, e] = input("Number of vertices and edges, separated by space: ").split(' ')
            graph = random_graph(int(v), int(e))
        elif option == "13":
            # values for tests: 1, 7 ;    2,5 -> no path;    5,8 ->2 edges
            [v1, v2] = input("The first and second vertex: ").split(' ')
            path = shortest_path_2(graph, int(v1), int(v2))
            print(path)
        elif option == "14":
            vertex_1 = int(input("Introduce the first vertex: "))
            vertex_2 = int(input("Introduce the second vertex ( the destination ) : "))
            graph.change_sign()
            print(dijkstra_lowest_cost_path(graph, vertex_1, vertex_2))
        elif option == "15":
            is_dag, toposort_path = tarjan_toposort(graph)
            if is_dag is True:
                path = highest_cost_path(graph)
                print(path)
        elif option == "0":
            done = True
            print("The program has been closed.")
        else:
            print("Invalid option!!!")

        graph_2 = graph.create_copy()
        write_to_file("copied_graph.txt", graph_2)


run()
