#!/usr/bin/env python

import sys
import re
import time
import heapq
import argparse
from os import system, name

class Ft_colors:
    PURPLE = '\x1b[94m'
    OKBLUE = '\x1b[96m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    UNDERLINE = '\x1b[4m'

class Node:
    count = 0
    def __init__(self, lst, id_parent, id, g, h, level):
        self.lst = lst
        self.lst_hash = hash(tuple(lst))
        self.id = id
        self.id_parent = id_parent
        self.f = g + h
        self.g = g
        self.level = level

    def __cmp__(self, other):
        return cmp(self.f, other.f)

    # generate child according to cost :
    # cost == -1 => No Heuristic function (Uniform Cost Search)
    # cost == 0 => No Weight consideration (Greedy Search)

    def generate_child(self, size, nbr_size, heuristic, cost):
        index = self.lst.index(0)
        x, y = index % size, index / size
        children = []
        val_list = [[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y]]
        for i in val_list:
            child = self.generate_items(self.lst, x, y, i[0], i[1], size)
            if child is not None:
                Node.count += 1
                g, h = self.g + 1, 0
                if cost != -1:
                    function = Heuristic(goal, child, size)
                    h = function.function[heuristic](nbr_size)
                if cost == 0:
                    g = 0
                children.append(Node(child, self.id, Node.count, g, h, self.level + 1))
        return children

    # generate different possible puzzle according to moves

    def generate_items(self, puz, x1, y1, x2, y2, size):
        if x2 >= 0 and x2 < size and y2 >= 0 and y2 < size:
            temp_puz = puz[:]
            temp_puz[x1 + y1 * size] = temp_puz[x2 + y2 * size]
            temp_puz[x2 + y2 * size] = 0
            return temp_puz
        return None

class Heuristic:
    def __init__(self, goal, current, size):
        self.goal = goal
        self.current = current
        self.size = size
        # this variable is a list of the functions name
        self.function = {'Euclidean_distance': self.ft_distance_euclide, 'manhattan': self.ft_manhattan, 'conflicts': self.ft_conflict, 'out_of_row_and_column': self.ft_out_of_row_and_column, 'hamming': self.ft_hamming}

    def ft_distance_euclide(self, nbr_index):
        dis = 0
        for key, element in enumerate(self.current):
            if element != 0:
                key_g = self.goal.index(element)
                x_n, y_n = key % self.size, key / self.size
                x_g, y_g = key_g % self.size, key_g / self.size
                dis += ((x_g - x_n) ** 2 + (y_g - y_n) ** 2) ** 0.5
        return dis

    def ft_out_of_row_and_column(self, nbr_index):
        count = 0
        for key, element in enumerate(self.current):
            if element != 0:
                key_g = self.goal.index(element)
                x_n, y_n = key % self.size, key / self.size
                x_g, y_g = key_g % self.size, key_g / self.size
                if x_g != x_n:
                    count += 1
                if y_n != y_g:
                    count += 1
        return count

    def ft_manhattan(self, nbr_index):
        count = 0
        for key, element in enumerate(self.current):
            if element != 0:
                index = self.goal.index(element)
                x_el, y_el = key % self.size, key / self.size
                x_goal, y_goal = index % self.size, index / self.size
                count += abs(x_el - x_goal) + abs(y_goal - y_el)
        return count

    def ft_hamming(self, nbr_index):
        count = 0
        for key, element in enumerate(self.current):
            if element != 0 and element != self.goal[key]:
                count += 1
        return count

    def ft_conflict(self, nbr_index):
        count = 0
        key = 0
        len_current = self.size ** 2
        while key < len_current:
            if self.current[key] == 0:
                key += 1
                continue
            key_g = nbr_index[self.current[key]]
            x_n, y_n = key % self.size, key / self.size
            x_g, y_g = key_g % self.size, key_g / self.size
            if y_g == y_n and x_n != x_g:
                limit_g = (key / size) * size
                limit_n = size * (key / size + 1)
                index_1 = key_g - 1
                while index_1 >= limit_g:
                    index_2 = key + 1
                    while index_2 < limit_n:
                        if self.goal[index_1] == self.current[index_2]:
                            count += 2
                            break
                        index_2 += 1
                    index_1 -= 1
            if x_g == x_n and y_n != y_g:
                index_1 = key_g - self.size
                while index_1 >= 0:
                    index_2 = key + self.size
                    while index_2 < len_current:
                        if self.goal[index_1] == self.current[index_2]:
                            count += 2
                            break
                        index_2 += self.size
                    index_1 -= self.size
            key += 1        
            count += abs(x_n - x_g) + abs(y_g - y_n)
        return count

class Puzzle:
    def __init__(self, size):
        self.n = size
        self.open = []
        self.closed = []
        self.tmp_closed = {}
        self.tmp_opened = {}

    def ft_find(self, child, puzzle_lst):
        index = 0
        for x in puzzle_lst:
            if x.lst_hash == child.lst_hash:
                return x, index
            index += 1
        return 0, index

    def ft_find_parent(self, id, puzzle_lst):
        for x in puzzle_lst:
            if x.id == id:
                return x
        return 0

    def ft_path(self, puzzle_lst):
        graph = []
        cur = puzzle_lst[0]
        while cur.id_parent > -1:
            graph.append(cur)
            cur = self.ft_find_parent(cur.id_parent, puzzle_lst)
        graph.append(cur)
        return graph

    def ft_print(self, graph):
        bcolors = Ft_colors()
        w = len(str(self.n * self.n))
        i, length_graph = 0, len(graph)
        while i < length_graph:
            if i != 0:
                print ""
                print("  | ")
                print("  | ")
                print(" \\\'/ \n")
            for key, x in enumerate(graph[i].lst):
                if x == 0:
                    print bcolors.FAIL + "0".rjust(w) + bcolors.ENDC,
                else:
                    print str(x).rjust(w),
                if (key + 1) % self.n == 0:
                    print ""
            i += 1

    # display different informations of the search   
    def ft_display(self, nb_open, nb_move, args, goal, start, time_exec):
        bcolors = Ft_colors()
        print bcolors.OKBLUE + "Greedy search:" + bcolors.ENDC, bcolors.OKGREEN + "YES" + bcolors.ENDC if args.g else bcolors.FAIL + "NO" + bcolors.ENDC
        print bcolors.OKBLUE + "Uniform cost search:" + bcolors.ENDC, bcolors.OKGREEN + "YES" + bcolors.ENDC if args.u else bcolors.FAIL + "NO" + bcolors.ENDC
        print bcolors.OKBLUE + "Solvable:" + bcolors.ENDC, bcolors.OKGREEN + "YES" + bcolors.ENDC
        print bcolors.WARNING + "Heuristic function:" + bcolors.ENDC, bcolors.UNDERLINE + args.f + bcolors.ENDC
        print bcolors.WARNING + "Puzzle size:" + bcolors.ENDC, bcolors.UNDERLINE + str(self.n) + bcolors.ENDC
        print bcolors.WARNING + "Goal type:" + bcolors.ENDC, bcolors.UNDERLINE + str(args.s) + bcolors.ENDC
        print bcolors.WARNING + "Initial state:" + bcolors.ENDC, bcolors.UNDERLINE + str(start) + bcolors.ENDC
        print bcolors.WARNING + "Goal state:" + bcolors.ENDC, bcolors.UNDERLINE + str(goal) + bcolors.ENDC
        if args.u:
            algo = "Uniform cost search"
        elif args.g:
            algo = "Greedy search"
        elif args.ida:
            algo = "IDA*"
        else:
            algo = 'A*'

        print bcolors.PURPLE + "Search algorithm:" + bcolors.ENDC, algo
        print bcolors.PURPLE + "Search duration:" + bcolors.ENDC, str(time_exec) + " seconds"
        print bcolors.PURPLE + "Evaluated nodes:" + bcolors.ENDC, str(nb_open)
        print bcolors.PURPLE + "Complexity in time:" + bcolors.ENDC,str(time_exec / nb_open) + " second(s) per node"
        print bcolors.PURPLE + "Number of moves:" + bcolors.ENDC, str(nb_move)
        print bcolors.PURPLE + "Space complexity:" + bcolors.ENDC, str(Node.count)
        print bcolors.FAIL + "Graph of solution:" + bcolors.ENDC
        if args.ida:
            # determine the graph from the inverse of open list
            graph = self.ft_path(self.open)[::-1]
        else:
            # determine the graph from the inverse of closed list
            graph = self.ft_path(self.closed)[::-1]
        if args.d:
            self.ft_print(graph)
        else:
            for i in graph:
                print i.lst
    
    # A* Algorithm
    def ft_astar(self, start, goal, heuristic, cost):
        nb_open = 0
        start = Node(start, -1, 0, 0, 0, 0)
        heapq.heappush(self.open, start)
        nb_open += 1
        self.tmp_opened[start.lst_hash] = start

        nbr_index = {}
        for key, element in enumerate(goal):
            nbr_index[element] = key
        while True:
            cur = heapq.heappop(self.open)
            del self.tmp_opened[cur.lst_hash]

            if cur.lst == goal:
                self.closed.insert(0, cur)
                return nb_open, cur.level

            self.closed.insert(0, cur)
            self.tmp_closed[cur.lst_hash] = cur
            children = cur.generate_child(size, nbr_index, heuristic, cost)
            for child in children:
                if child.lst_hash in self.tmp_closed:
                    continue
                if child.lst_hash not in self.tmp_opened:
                    self.tmp_opened[child.lst_hash] = child
                    heapq.heappush(self.open, child)
                    nb_open += 1
                else:
                    actual_node, index = self.ft_find(child, self.open)
                    if actual_node.g > child.g:
                        self.tmp_opened[child.lst_hash] = child
                        del self.open[index]
                        heapq.heappush(self.open, child)
        return 0, 0

    # IDA* Algorithm
    def ft_idastar(self, start, goal, size, heuristic):
        nb_open = 0
        threshold = start.f
        dic = {}
        nbr_index = {}
        for key, element in enumerate(goal):
            nbr_index[element] = key
        self.open.insert(0, start)
        nb_open += 1
        self.tmp_opened[start.lst_hash] = start
        while True:
            temp, nb_open = self.ft_search(threshold, goal, size, dic, nbr_index, nb_open, heuristic)
            if temp == 0:
                return nb_open, self.open[0].level
            threshold = temp
        return 0, 0

    # Recursive function ft_search for IDA* Algorithm 
    def ft_search(self, threshold, goal, size, dic, nbr_index, nb_open, heuristic):
        current = self.open[0]
        f = current.f
        if f > threshold:
            return f, nb_open
        if current.lst_hash == hash(tuple(goal)):
            return 0, nb_open
        min = sys.maxint

        if current.id not in dic:
            dic[current.id] = current.generate_child(size, nbr_index, heuristic, 2)

        for child in dic[current.id]:
            if child.lst_hash not in self.tmp_opened:
                self.open.insert(0, child)
                nb_open += 1
                self.tmp_opened[child.lst_hash] = child
                temp, nb_open = self.ft_search(threshold, goal, size, dic, nbr_index, nb_open, heuristic)
                if temp == 0:
                    return 0, nb_open
                if temp < min:
                    min = temp
                del self.tmp_opened[self.open[0].lst_hash]
                del self.open[0]
        return min, nb_open

# Goal in snail
def ft_spiralPrint(n):
    k, l = 0, 0
    m, h = n, n
    size, d = 1, n ** 2
    tab = [[0 for i in range(n)] for j in range(n)]

    while (k < m and l < n):
        for i in range(l, n):
            tab[k][i] = size % d
            size += 1

        k += 1
        for i in range(k, m):
            tab[i][n - 1] = size % d
            size += 1

        n -= 1
        if (k < m):

            for i in range(n - 1, (l - 1), -1):
                tab[m - 1][i] = size % d
                size += 1
            m -= 1

        if (l < n):
            for i in range(m - 1, k - 1, -1):
                tab[i][l] = size % d
                size += 1
            l += 1

    goal = [tab[i][j] for i in range(h) for j in range(h)]

    return goal

# Goal: zero_last
def ft_zero_last(size):
    goal = [i + 1 for i in range(0, (size ** 2) - 1)]
    goal.append(0)
    return goal

# Goal : zero_first
def ft_zero_first(size):
    goal = [i for i in range(0, size ** 2)]
    return goal

def ft_atoi(string):
    res = 0
    for i in xrange(len(string)):
        res = res * 10 + (ord(string[i]) - ord('0'))
    return res

# generate a puzzle from a str 
def generate_Puzzle(lst_str, size):
    i = 0
    lst = []
    size = size ** 2
    while i < size:
        lst.append(ft_atoi(lst_str[i]))
        i += 1
    return lst

# determine if the puzzle is solvent or not before applying algorithm
def ft_solvable(start, goal, size):
    xstart, ystart = start.index(0) % size, start.index(0) / size
    # array goal
    xgoal, ygoal = goal.index(0) % size, goal.index(0) / size
    # moves
    dep = abs(ystart - ygoal) + abs(xstart - xgoal)
    # calcul nb inverstion
    nb = 0
    i = 0
    size = size ** 2
    while i < size:
        nb = nb + len(list(set(start[i:]) - set(goal[goal.index(start[i]):])))
        i += 1

    sol = 0
    if (nb % 2 == 0 and dep % 2 == 0) or (nb % 2 != 0 and dep % 2 != 0):
        sol = 1
    return sol

# save history of statistics in the file 'N-Puzzle_statistics' 
def ft_save_history(start, goal, size, nb_open, end_time):
    with open('N-Puzzle_statistics', 'ab') as f:
        f.write(str(size) + ';' + str(goal) + ';' + str(start) + ';' + str(Node.count) + ';' + str(nb_open) + ';' + str(end_time) + '\n')

# display history of statistics saved in the file 'N-Puzzle_statistics'
def ft_display_history():
    try:
        print "\x1b[91m" + "\nList of History" + "\x1b[0m"
        with open('N-Puzzle_statistics') as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        for i in range(0, 30):
            print '-',
        print ""

        for k, elem in enumerate(content):
            line = content[k].split(';')
            size = ft_atoi(line[0])
            goal = list(filter(None, re.split(r',| |\[|\]', line[1])))
            start = list(filter(None, re.split(r',| |\[|\]', line[2])))
            level = line[4]
            end_time = round(float(line[5]), 3)

            bcolors = Ft_colors()
            w = len(str(size * size))
            i, j = 0, 0
            for key, x in enumerate(start):
                print str(x).rjust(w),
                if (key + 1) % size == 0:
                    print '\t\t',
                    index = key - (size - 1)
                    while index < size ** 2:
                        print str(goal[index]).rjust(w),
                        if (index + 1) % size == 0:
                            if j == 0: print bcolors.OKBLUE + "\t\tPuzzle size:" + bcolors.ENDC, bcolors.OKGREEN + str(size) + bcolors.ENDC
                            if j == 1: print bcolors.OKBLUE + "\t\tNumber of moves:" + bcolors.ENDC, bcolors.OKGREEN + str(level) + bcolors.ENDC
                            if j == 2: print bcolors.OKBLUE + "\t\tSearch duration:" + bcolors.ENDC, bcolors.OKGREEN + str(end_time) + bcolors.ENDC
                            j += 1
                            break
                        index += 1
                    if j > 3:
                        print ""
            if size != 3:
                print ""
            for i in range(0, size*2 + 30):
                print '-',
            print ""
    except:
        print 'No History'

# clear the terminal every time we display final results 
def ft_clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

# Main Function
if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description='n-puzzle @ 42')
    parser.add_argument("-ida", action='store_true', help="ida* search")
    parser.add_argument("-u", action='store_true', help="uniform-cost search")
    parser.add_argument("-g", action='store_true', help="greedy search")
    parser.add_argument("-f", choices=['hamming', 'manhattan', 'conflicts', 'Euclidean_distance',
                                       'out_of_row_and_column'], default='manhattan', help="heuristic function")
    parser.add_argument("-s", choices=['zero_first', 'zero_last', 'snail'], default='snail', help="solved state")
    parser.add_argument("-d", action='store_true', help="display graph")
    parser.add_argument("-hist", action='store_true', help="display per-puzzle statistics")
    parser.add_argument("input", help="input start")
    parser.parse_args()
    args = parser.parse_args()

    data = args.input
    
    # start list
    start = list(filter(None, re.split(r'# This puzzle is solvable|# This puzzle is unsolvable|\n|,| |', data)))
    
    # size
    size = len(start) ** 0.5
    if int(size) != size:
        size = ft_atoi(start[0])
        start = start[1:]
        if size ** 2 != len(start):
            print("ERROR : Puzzle size")
            exit()
    size = int(size)

    # generate puzzle
    start = generate_Puzzle(start, size)

    # goals
    if args.s == 'zero_first':
        goal = ft_zero_first(size)
    elif args.s == 'zero_last':
        goal = ft_zero_last(size)
    else:
        goal = ft_spiralPrint(size)
    sol = ft_solvable(start, goal, size)
    if sol == 1:
        cur = Puzzle(size)
        start_time = time.time()
        print "Wait please..."

        # apply the right algorithm according to flags
        if args.ida:
            root = Node(start, -1, 0, 0, 0, 0)
            nb_open, level = cur.ft_idastar(root, goal, size, args.f)
        else:
            if args.u:
                nb_open, level = cur.ft_astar(start, goal, args.f, -1)
            elif args.g:
                nb_open, level = cur.ft_astar(start, goal, args.f, 0)
            else:
                nb_open, level = cur.ft_astar(start, goal, args.f, 2)
        end_time = time.time() - start_time
        ft_clear()
        ft_save_history(start, goal, size, level, end_time)
        cur.ft_display(nb_open, level, args, goal, start, end_time)
        if args.hist:
            ft_display_history()
    else:
        print "\x1b[91m" + "This puzzle is unsolvable" + "\x1b[0m"