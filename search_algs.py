from Graph import Graph
from collections import defaultdict
from collections import deque
from random import randint
from random import choice
from random import sample
import operator
import numpy as np
import string
import json
import pprint
import time

from queue import PriorityQueue
#from Queue import PriorityQueue

# Global vars
World = np.zeros([10, 10], dtype=np.int)
CITY_DICT = defaultdict(dict)


AVG_BRANCHES = 0
BRANCHES_GENERATED = 0
NODES_VISITED_BFS = 0
NODES_VISITED_DFS = 0
NODES_VISITED_IDDFS = 0
NODES_VISITED_ASTAR = 0

MAX_NODES_GEN_BFS = 0
MAX_NODES_GEN_DFS = 0
MAX_NODES_GEN_IDDFS = 0

MAX_NODES_GEN_ASTAR = 0




def create_astar_path(parent, start, end):
    curr_city = end
    path = []
    while curr_city != None:
        path.append(curr_city)
        curr_city = parent[curr_city]
    return path[::-1]

def a_star(graph, start, end, heuristic):
    PQ = PriorityQueue()
    PQ.put((0, start))
    parent = {start: None}
    # Initial cost is 0
    cost = {start: 0}

    count = 0
    while not PQ.empty():
        count += 1

        node = PQ.get()

        global NODES_VISITED_ASTAR
        NODES_VISITED_ASTAR += 1

        if node == end:
            break

        else:
            neighbors = graph.return_edges(node[1])
            for neighbor in neighbors:
                # Cost from city to city is still euclidean distance
                next_cost = cost[node[1]] + euclidian_distance(node[1], neighbor)
                if neighbor not in cost or next_cost < cost[neighbor]:
                    cost[neighbor] = next_cost
                    # Save the neighbor and heuristic val in PriorityQueue
                    heuristic_func = next_cost + heuristic(neighbor, end)
                    PQ.put((heuristic_func, neighbor))
                    parent[neighbor] = node[1]

    return create_astar_path(parent, start, end)


def DLS(graph, path, depth, end_node, visited):
    global NODES_VISITED_IDDFS
    NODES_VISITED_IDDFS += 1

    visited.append(path[-1])

    if path[-1] == end_node:
        return path

    if depth <= 0:
        return None

    neighbors = graph.return_edges(path[-1])
    for child in neighbors:
        if child not in path:
            next_path = DLS(graph, path + [child], depth-1, end_node, visited)
            if next_path:
                return next_path



def IDDFS(graph, root, end_node):
    for N in range(25):
        visited = []
        path = DLS(graph, [root], N, end_node, visited)
        if path:
            return path

    return False



def dfs_helper(graph, path, end, visited):
    global NODES_VISITED_DFS
    NODES_VISITED_DFS += 1

    visited.append(path[-1])

    if path[-1] == end:
        return path

    neighbors = graph.return_edges(path[-1])
    for neighbor in neighbors:
        if neighbor not in visited:

            new_path = dfs_helper(graph, path + [neighbor], end, visited)
            if new_path:
                return new_path

    return None

def dfs_search(graph, start, end):
    visited = []
    return dfs_helper(graph, [start], end, visited)


def bfs_search(graph, start, end):
    # container like a queue, allows to pop from left or right
    q = [[start]]
    explored = []

    if start == end:
        return

    while q:
        path = q.pop(0)
        explored.append(path[-1])

        global NODES_VISITED_BFS
        NODES_VISITED_BFS += 1

        neighbors = graph.return_edges(path[-1])

        for neighbor in neighbors:
            if neighbor not in explored:
                new_path = list(path)
                new_path.append(neighbor)
                q.append(new_path)

                if neighbor == end:
                    return new_path

    return None



def constant_heuristic(a, b):
    return 0

def manhattan_distance(a, b):
    x = abs(CITY_DICT[a]['location'][0] - CITY_DICT[b]['location'][0])
    y = abs(CITY_DICT[a]['location'][1] - CITY_DICT[b]['location'][1])
    val = x + y
    return val

def euclidian_distance(a, b):
    x = abs(CITY_DICT[a]['location'][0] - CITY_DICT[b]['location'][0])
    y = abs(CITY_DICT[a]['location'][1] - CITY_DICT[b]['location'][1])
    euclidian_distance = (x**2 + y**2)**(1/2)
    return euclidian_distance

# neighbors will be chosen randomly from closest 5 cities
def gen_neighbors():

    for city in CITY_DICT:
        CITY_DICT[city]['neighbors'] = set()

    for city in CITY_DICT:
        num_branches = randint(1, 4)
        closest = gen_closest_cities(city)
        neighbor_cities = sample(closest, num_branches)

        for neigh in neighbor_cities:
            CITY_DICT[city]['neighbors'].add(neigh[0])
            CITY_DICT[neigh[0]]['neighbors'].add(city)

    return


def gen_cities():
    for i in range(0, 26):
        CITY_DICT[i]['location'] = (randint(0, 9), randint(0, 9))
        World[CITY_DICT[i]['location']] = 1
    return


# Returns 5 closest cities to a city
def gen_closest_cities(city):

    CITY_DICT[city]['closest_cities'] = {}
    for a in CITY_DICT:
        CITY_DICT[city]['closest_cities'][a] = euclidian_distance(city, a)

    closest_cities = CITY_DICT[city]['closest_cities']
    sorted_vals = sorted(closest_cities.items(), key=operator.itemgetter(1), reverse=True)
    closest5 = sorted_vals[1:6]

    return closest5

def get_num_branches():
    branches = 0
    for city in CITY_DICT:
        branches += len(CITY_DICT[city]['neighbors'])

    return branches


if __name__ == "__main__":
    COUNT = 100

    AVG_BRANCHES = 0
    AVG_RUNTIME_BFS = 0
    AVG_RUNTIME_DFS = 0
    AVG_RUNTIME_IDDFS = 0
    AVG_RUNTIME_ASTAR_EUCLIDIAN = 0
    AVG_RUNTIME_ASTAR_MANHATTAN = 0
    AVG_RUNTIME_ASTAR_CONSTANT = 0


    AVG_NODES_VISITED_ASTAR_EUCLIDIAN = 0
    AVG_NODES_VISITED_ASTAR_MANHATTAN = 0
    AVG_NODES_VISITED_ASTAR_CONSTANT = 0



    print("Starting tests... \n")

    for i in range(COUNT):
        # reset all variables

        # generate list of 26 randomly located cities
        gen_cities()
        gen_neighbors()
        num_branches = get_num_branches()
        BRANCHES_GENERATED += num_branches

        # generate start city and unique end city
        start_city = randint(0, 25)
        end_city = start_city
        while end_city == start_city:
            end_city = randint(0, 25)

        adjacency_list = {}
        for city in CITY_DICT:
            adjacency_list[city] = CITY_DICT[city]['neighbors']

        graph = Graph(adjacency_list)


        print("Start City: ", start_city)
        print("End City:   ", end_city)

        start_time = time.time()
        print("BFS:    ", bfs_search(graph, start_city, end_city))
        end_time = time.time()
        AVG_RUNTIME_BFS += (end_time - start_time)

        start_time = time.time()
        print("DFS:    ", dfs_search(graph, start_city, end_city))
        end_time = time.time()
        AVG_RUNTIME_DFS += (end_time - start_time)

        start_time = time.time()
        print("IDDFS:  ", IDDFS(graph, start_city, end_city))
        end_time = time.time()
        AVG_RUNTIME_IDDFS += (end_time - start_time)


        start_time = time.time()
        print("ASTAR_EUCLIDIAN:  ", a_star(graph, start_city, end_city, euclidian_distance))
        end_time = time.time()
        AVG_RUNTIME_ASTAR_EUCLIDIAN += (end_time - start_time)
        AVG_NODES_VISITED_ASTAR_EUCLIDIAN += NODES_VISITED_ASTAR
        NODES_VISITED_ASTAR = 0

        start_time = time.time()
        print("ASTAR_MANHATTAN:  ", a_star(graph, start_city, end_city, manhattan_distance))
        end_time = time.time()
        AVG_RUNTIME_ASTAR_MANHATTAN += (end_time - start_time)
        AVG_NODES_VISITED_ASTAR_MANHATTAN += NODES_VISITED_ASTAR
        NODES_VISITED_ASTAR = 0

        start_time = time.time()
        print("ASTAR_CONSTANT:   ", a_star(graph, start_city, end_city, constant_heuristic))
        end_time = time.time()
        AVG_RUNTIME_ASTAR_CONSTANT += (end_time - start_time)
        AVG_NODES_VISITED_ASTAR_CONSTANT += NODES_VISITED_ASTAR
        NODES_VISITED_ASTAR = 0


    # STATISTICS

    # Average space complexity (maximum number of nodes generated)

    # Average time complexity (number of nodes visited)

    # Actual running time (in seconds)

    # Average path length

    # Number of problems solved


    AVG_BRANCHES = BRANCHES_GENERATED/COUNT
    AVG_NODES_VISITED_BFS = NODES_VISITED_BFS/COUNT
    AVG_NODES_VISITED_DFS = NODES_VISITED_DFS/COUNT
    AVG_NODES_VISITED_IDDFS = NODES_VISITED_IDDFS/COUNT

    AVG_NODES_VISITED_ASTAR_EUCLIDIAN = AVG_NODES_VISITED_ASTAR_EUCLIDIAN/COUNT
    AVG_NODES_VISITED_ASTAR_MANHATTAN = AVG_NODES_VISITED_ASTAR_MANHATTAN/COUNT
    AVG_NODES_VISITED_ASTAR_CONSTANT = AVG_NODES_VISITED_ASTAR_CONSTANT/COUNT




    AVG_RUNTIME_BFS = (AVG_RUNTIME_BFS/COUNT)
    AVG_RUNTIME_DFS = (AVG_RUNTIME_DFS/COUNT)
    AVG_RUNTIME_IDDFS = (AVG_RUNTIME_IDDFS/COUNT)
    AVG_RUNTIME_ASTAR_EUCLIDIAN = (AVG_RUNTIME_ASTAR_EUCLIDIAN/COUNT)
    AVG_RUNTIME_ASTAR_MANHATTAN = (AVG_RUNTIME_ASTAR_MANHATTAN/COUNT)
    AVG_RUNTIME_ASTAR_CONSTANT = (AVG_RUNTIME_ASTAR_CONSTANT/COUNT)




    print("AVG_BRANCHES:        ", AVG_BRANCHES)
    print("AVG_NODES_VISITED_BFS:   ", AVG_NODES_VISITED_BFS)
    print("AVG RUNTIME BFS:         ", AVG_RUNTIME_BFS)

    print("AVG_NODES_VISITED_DFS:   ", AVG_NODES_VISITED_DFS)
    print("AVG RUNTIME DFS:         ", AVG_RUNTIME_DFS)

    print("AVG_NODES_VISITED_IDDFS: ", AVG_NODES_VISITED_IDDFS)
    print("AVG RUNTIME IDDFS:       ", AVG_RUNTIME_IDDFS)


    print()
    print("AVG_NODES_VISITED_ASTAR_EUCLIDIAN: ", AVG_NODES_VISITED_ASTAR_EUCLIDIAN)
    print("AVG RUNTIME ASTAR:                 ", AVG_RUNTIME_ASTAR_EUCLIDIAN)
    print()

    print("AVG_NODES_VISITED_ASTAR_MANHATTAN: ", AVG_NODES_VISITED_ASTAR_MANHATTAN)
    print("AVG RUNTIME ASTAR:                 ", AVG_RUNTIME_ASTAR_MANHATTAN)
    print()

    print("AVG_NODES_VISITED_ASTAR_CONSTANT:  ", AVG_NODES_VISITED_ASTAR_CONSTANT)
    print("AVG RUNTIME ASTAR:                 ", AVG_RUNTIME_ASTAR_CONSTANT)
    print()
