import sys
import utils
import random
from search import *
import copy
import time

def all(iter):
    for e in iter:
        if not e: return False
    return True

def any(iter):
    for e in iter:
        if e: return True
    return False


# takes multi-line map string, trims indentation, replaces newlines with given separator
def format_map_str(tiles,sep):
    return sep.join(line.strip() for line in tiles.splitlines())

class Map:
    def __init__(self,w,h,tile_str=None):

        if tile_str is None:
            # just create a clear map
            self.tiles = []
            self.w = w
            self.h = h
            for i in range(w*h):
                self.tiles.append('.')
        else:
            self.setMap(w,h,tile_str)

        # sets logging verbosity (on|off)
        self.verbose = False

    # create a map from a tile string
    def setMap(self,w,h,tile_str):
        self.w = w
        self.h = h
        self.tiles = list(format_map_str(tile_str,""))

    # creates a string of the current map
    def __str__(self):
        s = "\n"
        i = 0
        for y in range(self.h):
            for x in range(self.w):
                s += self.tiles[i]
                i += 1
            s += "\n"
        return s

    # converts x,y to index
    def xy_to_i(self,x,y):
        return x+y*self.w

    # converts index to x,y
    def i_to_xy(self,i):
        return i%self.w, i/self.w

    # validates x,y
    def xy_valid(self,x,y):
        return x >= 0 and x < self.w and y>=0 and y<self.h

    # gets tile at x,y or returns None if invalid
    def get_tile(self,x,y):
        if not self.xy_valid(x,y):
            return None
        return self.tiles[x+y*self.w]

    # adds a single wall tile at x,y
    def add_wall_tile(self,x,y):
        if self.xy_valid(x,y):
            self.tiles[x+y*self.w] = '|'

    def is_wall_block_filled(self,x,y):
        return all(self.get_tile(x+dx,y+dy) == '|' for dy in range(1,3) for dx in range(1,3))

    # adds a 2x2 block inside the 4x4 block at the given x,y coordinate
    def add_wall_block(self,x,y):
        self.add_wall_tile(x+1,y+1)
        self.add_wall_tile(x+2,y+1)
        self.add_wall_tile(x+1,y+2)
        self.add_wall_tile(x+2,y+2)

    # determines if a 2x2 block can fit inside the 4x4 block at the given x,y coordinate
    # (the whole 4x4 block must be empty)
    def can_new_block_fit(self,x,y):
        if not (self.xy_valid(x,y) and self.xy_valid(x+3,y+3)):
            return False
        for y0 in range(y,y+4):
            for x0 in range(x,x+4):
                if self.get_tile(x0,y0) != '.':
                    return False
        return True

    # create a list of valid starting positions
    def update_pos_list(self):
        self.pos_list = []
        for y in range(self.h):
            for x in range(self.w):
                if self.can_new_block_fit(x,y):
                    self.pos_list.append((x,y))

    # A connection is a sort of dependency of one tile block on another.
    # If a valid starting position is against another wall, then add this tile
    # to other valid start positions' that intersect this one so that they fill
    # it when they are chosen.  This filling is a heuristic to eliminate gaps.
    def update_connections(self):
        self.connections = {}
        for y in range(self.h):
            for x in range(self.w):
                if (x,y) in self.pos_list:
                    if any(self.get_tile(x-1,y+y0)=='|' for y0 in range(4)): self.add_connection(x,y,1,0)
                    if any(self.get_tile(x+4,y+y0)=='|' for y0 in range(4)): self.add_connection(x,y,-1,0)
                    if any(self.get_tile(x+x0,y-1)=='|' for x0 in range(4)): self.add_connection(x,y,0,1)
                    if any(self.get_tile(x+x0,y+4)=='|' for x0 in range(4)): self.add_connection(x,y,0,-1)

    # the block at x,y is against a wall, so make intersecting blocks in the direction of
    # dx,dy fill the block at x,y if they are filled first.
    def add_connection(self,x,y,dx,dy):
        def connect(x0,y0):
            src = (x,y)
            dest = (x0,y0)
            if not dest in self.pos_list:
                return
            if dest in self.connections:
                self.connections[dest].append(src)
            else:
                self.connections[dest] = [src]
        if (x,y) in self.pos_list:
            connect(x+dx,y+dy)
            connect(x+2*dx,y+2*dy)
            if not (x-dy,y-dx) in self.pos_list: connect(x+dx-dy,y+dy-dx)
            if not (x+dy,y+dx) in self.pos_list: connect(x+dx+dy,y+dy+dx)
            if not (x+dx-dy,y+dy-dx) in self.pos_list: connect(x+2*dx-dy, y+2*dy-dx)
            if not (x+dx+dy,y+dy+dx) in self.pos_list: connect(x+2*dx+dy, y+2*dy+dx)

    # update the starting positions and dependencies
    def update(self):
        self.update_pos_list()
        self.update_connections()

    # expand a wall block at the given x,y
    # return number of tiles added
    def expand_wall(self,x,y):
        visited = []
        def expand(x,y):
            count = 0
            src = (x,y)
            if src in visited:
                return 0
            visited.append(src)
            if src in self.connections:
                for x0,y0 in self.connections[src]:
                    if not self.is_wall_block_filled(x0,y0):
                        count += 1
                        self.add_wall_block(x0,y0)
                    count += expand(x0,y0)
            return count
        return expand(x,y)

    def get_most_open_dir(self,x,y):
        dirs = ((0,-1),(0,1),(1,0),(-1,0))
        max_dir = random.choice(dirs)
        max_len = 0
        for dx,dy in dirs:
            len = 0
            while (x+dx*len,y+dy*len) in self.pos_list:
                len += 1
            if len > max_len:
                max_dir = (dx,dy)
                max_len = len
        return max_dir

    # start a wall at block x,y
    def add_wall_obstacle(self,x=None,y=None,extend=False):
        self.update()
        if not self.pos_list:
            return False

        # choose random valid starting position if none provided
        if (x is None or y is None):
            x,y = random.choice(self.pos_list)

        # add first block
        self.add_wall_block(x,y)

        # initialize verbose print lines
        first_lines = str(self).splitlines()
        grow_lines = [""]*(self.h+2)
        extend_lines = [""]*(self.h+2)

        # mandatory grow phase
        count = self.expand_wall(x,y)
        if count > 0:
            grow_lines = str(self).splitlines()

        # extend phase
        if extend:

            # desired maximum block size
            max_blocks = 4

            # 35% chance of forcing the block to turn
            # turn means the turn has been taken
            # turn_blocks is the number of blocks traveled before turning
            turn = False
            turn_blocks = max_blocks
            if random.random() <= 0.35:
                turn_blocks = 4
                max_blocks += turn_blocks

            # choose a random direction
            dx,dy = random.choice(((0,-1),(0,1),(1,0),(-1,0)))
            orig_dir = (dx,dy)

            i = 0
            while count < max_blocks:
                x0 = x+dx*i
                y0 = y+dy*i
                # turn if we're past turning point or at a dead end
                if (not turn and count >= turn_blocks) or not (x0,y0) in self.pos_list:
                    turn = True
                    dx,dy = -dy,dx # rotate
                    i = 1
                    # stop if we've come full circle
                    if orig_dir == (dx,dy): break
                    else: continue

                # add wall block and grow to fill gaps
                if not self.is_wall_block_filled(x0,y0):
                    self.add_wall_block(x0,y0)
                    count += 1 + self.expand_wall(x0,y0)
                i += 1
            extend_lines = str(self).splitlines()

        # print the map states after each phase for debugging
        if self.verbose:
            print("added block at %i,%i" % (x,y))
            for a,b,c in zip(first_lines, grow_lines, extend_lines):
                print(a,b,c)

        return True

def genMaze(width = 16, height = 31, debbug = False):

    #Estou assumindo que:
    # O espaço inicial dos fantasmas tem 8 de largura e 5 de altura (espaço do meio)
    # A altura e largura mínima do labirinto consiste em um espaço para percorrer em volta do espaço inicial
    # Portanto altura >= 9, largura >= 12. Para facilitar, a altura deve ser sempre ímpar.
    # Diferente do walls.py original, vou duplicar na vertical o valor de width e não as 14 primeiras linhas
    if (width < 12) or (height < 9) or ((height % 2) == 0):
        print('Size not allowed')
        return None

    wall = "|"*width + "\n"
    path = "|" + "."*(width-1) + "\n"
    middle = "|" + "."*(width-5) + "||||" + "\n"

    strMaze = wall + path*int(((height-7)/2)) + middle*5 + path*int(((height-7)/2)) + wall
    if debbug:
      print('Initial Maze:')
      print(strMaze)

    # initial empty map with standard ghost house
    tileMap = Map(width,height,strMaze)

    # verbosity option (-v)
    #if len(sys.argv) > 1 and sys.argv[1] == "-v":
        #tileMap.verbose = True
    tileMap.verbose = debbug

    # generate map by adding walls until there's no more room
    while tileMap.add_wall_obstacle(extend=True):
        pass

    # reflect the first 14 columns to print the map
    strMaze = [[] for i in range(height)]
    for i,line in enumerate(str(tileMap).splitlines()):
        if i > 0:
          s = line[:width]
          line = s+s[::-1]
          #strMaze += line + '\n'
          strMaze[i-1] = [x for x in line]
    return strMaze

def printMaze(maze):
  for line in maze:
    print(''.join(line))

def printResult(problem, result):
  print('Result found: %s' % (result[0]!=None))
  print('Execution time: %fs' % (result[1]))
  print('Nodes tested: %i' % (result[2]))
  print('Nodes expanded: %i' % (result[3]))
  print('Nodes to solution: %i' % (result[4]))
  if result[0]:
    for path in result[0].path():
      x,y = path.state
      problem.maze[y][x] = 'o'
  problem.maze[problem.initial[1]][problem.initial[0]] = 'P'
  problem.maze[problem.goal[1]][problem.goal[0]] = 'G'
  for line in problem.maze:
      print(''.join(line))
  if result[0]:
    for path in result[0].path():
      x,y = path.state
      problem.maze[y][x] = ' '
  problem.maze[problem.initial[1]][problem.initial[0]] = 'P'
  problem.maze[problem.goal[1]][problem.goal[0]] = 'G'
  return None

class mazeProblem(Problem):
  def __init__(self, initial, goal, maze):
    Problem.__init__(self, initial, goal)
    self.maze = copy.deepcopy(maze)
    self.invalid = ['|','A','B','C']
    self.states_visited = [initial]

  def actions(self, state):
    x,y = state
    possibleActions = []
    if self.maze[y][x+1] not in self.invalid and (x+1,y) not in self.states_visited: possibleActions.append((x+1,y)) #Right
    if self.maze[y][x-1] not in self.invalid and (x-1,y) not in self.states_visited: possibleActions.append((x-1,y)) #Left
    if self.maze[y+1][x] not in self.invalid and (x,y+1) not in self.states_visited: possibleActions.append((x,y+1)) #Up
    if self.maze[y-1][x] not in self.invalid and (x,y-1) not in self.states_visited: possibleActions.append((x,y-1)) #Down
    self.states_visited.extend(possibleActions)
    return possibleActions

  def result(self, state, action):
    #x,y = state
    #maze[y][x] = ';'
    return action

  def goal_test(self, state):
    #if self.maze[state[1]][state[0]] == 'G':
    if self.goal == state:
      return True
    else:
      self.maze[state[1]][state[0]] = ' '
      return False

  def path_cost(self, c, state1, action, state2):
    return c + 1

  def clean_visited_states(self):
    self.states_visited = []

class Node:
  def __init__(self, state, parent=None, action=None, path_cost=0):
    self.state = state
    self.parent = parent
    self.action = action
    self.path_cost = path_cost
    self.depth = 0
    if parent:
        self.depth = parent.depth + 1

  def __repr__(self):
    return "<Node {}>".format(self.state)

  def __lt__(self, node):
    return self.state < node.state

  def __eq__(self, other):
    return isinstance(other, Node) and self.state == other.state

  def __hash__(self):
    return hash(self.state)

  def expand(self, problem):
    return [self.child_node(problem, action) for action in problem.actions(self.state)]

  def child_node(self, problem, action):
    next_state = problem.result(self.state, action)
    next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
    return next_node

  def solution(self):
    """Return the sequence of actions to go from the root to this node."""
    return [node.action for node in self.path()[1:]]

  def path(self):
    """Return a list of nodes forming the path from the root to this node."""
    node, path_back = self, []
    while node:
        path_back.append(node)
        node = node.parent
    return list(reversed(path_back))

def breadth_first_tree_search(problem):
  start = time.time()
  #Adding first node to the queue
  frontier = deque([Node(problem.initial)])
  tested = 0
  while frontier:
    #Popping first node of queue
    node = frontier.popleft()
    tested += 1
    if problem.goal_test(node.state):
      return node, time.time()-start, tested, tested + len(frontier), len(node.path())

    frontier.extend(node.expand(problem))
  return None, time.time()-start, tested, tested + len(frontier), np.inf

def depth_first_tree_search(problem):
  start = time.time()
  #Adding first node to the queue
  frontier = [Node(problem.initial)]
  tested = 0
  while frontier:
    #Popping first node of queue
    node = frontier.pop()
    tested += 1
    if problem.goal_test(node.state):
      return node, time.time()-start, tested, tested + len(frontier), len(node.path())

    frontier.extend(node.expand(problem))

  return None, time.time()-start, tested, tested + len(frontier), np.inf

# Greedy Best First Search
def greedy_bestfirst_tree_search(problem, heuristic='m'):
  start = time.time()
  cache = {}

  def manhattan(n, goal):
    if n in cache:
      return cache[n]
    dist = np.abs(goal[0] - n.state[0]) + np.abs(goal[1] - n.state[1])
    cache[n] = dist
    return dist

  def euclidean(n, goal):
    if n in cache:
      return cache[n]
    dist = np.linalg.norm([goal[0] - n.state[0],goal[1] - n.state[1]])
    cache[n] = dist
    return dist

  def h(n, goal, heuristic):
    if heuristic == 'e':
      return euclidean(n, goal)
    else:
      return manhattan(n, goal)

  #Adding first node to the queue
  queue = [Node(problem.initial)]
  tested = 0
  while queue:
    #Popping first node of queue
    node = queue.pop(0)
    tested += 1
    if problem.goal_test(node.state):
      return node, time.time()-start, tested, tested + len(queue), len(node.path())

    #Expand first node in the queue
    expand_list = node.expand(problem)

    #Sort expanded nodes
    gn = np.argsort(np.asarray([h(n, problem.goal, heuristic) for n in expand_list]))
    expand_list = [expand_list[index] for index in gn]

    #Insert them in the queue
    i=0
    while len(expand_list) > 0 and i < len(queue):
      if h(expand_list[0], problem.goal, heuristic) < h(queue[i], problem.goal, heuristic):
        queue.insert(i, expand_list.pop(0))
      i+=1
    if len(expand_list) > 0:
      queue.extend(expand_list)

  return None, time.time()-start, tested, tested + len(queue), np.inf

def a_star_search(problem):
  start = time.time()
  cache = {}

  def cost(n, goal):
    if n in cache:
      return cache[n] + n.depth

    dist = np.linalg.norm([goal[0] - n.state[0],goal[1] - n.state[1]])
    cache[n] = dist
    return dist + n.depth

  #Adding first node to the queue
  queue = [Node(problem.initial)]
  tested = 0
  while queue:
    #print('----------------------')
    #for line in problem.maze:
    #  print(''.join(line))
    #Popping first node of queue
    node = queue.pop(0)
    tested += 1
    if problem.goal_test(node.state):
      return node, time.time()-start, tested, tested + len(queue), len(node.path())

    #Expand first node in the queue
    expand_list = node.expand(problem)

    #Sort expanded nodes
    gn = np.argsort(np.asarray([cost(n, problem.goal) for n in expand_list]))
    expand_list = [expand_list[index] for index in gn]

    #Insert them in the queue
    i=0
    while len(expand_list) > 0 and i < len(queue):
      if cost(expand_list[0], problem.goal) < cost(queue[i], problem.goal):
        queue.insert(i, expand_list.pop(0))
      i+=1
    if len(expand_list) > 0:
      queue.extend(expand_list)


  return None, time.time()-start, tested, tested + len(queue), np.inf

def h(node, problem):
  return np.abs(node.state[0]-problem.goal[0]) + np.abs(node.state[1]-problem.goal[1])

def aux_BFS(current, problem):

  problem.clean_visited_states()

  frontier = deque([current])
  value = h(current, problem)
  expanded = 0
  while frontier:
    #Popping first node of queue
    node = frontier.popleft()
    if h(node, problem) < value:
      return node, expanded + len(frontier)
    frontier.extend(node.expand(problem,))
    expanded += 1
  return None,  expanded +len(frontier)

def enforced_hill_climbing(problem):

  start = time.time()
  BFStested = 0
  tested = 0
  current = Node(problem.initial)
  neighbors = current.expand(problem)
  expanded = 1
  while (h(current, problem) != 0):
    next_state = current
    #Select lowest h among neighbors AND current state
    for neighbor in neighbors:
      if h(neighbor, problem) < h(next_state, problem):
        next_state = neighbor

    #Checks if it is the goal state
    tested += 1
    if problem.goal_test(next_state.state):
      return current, time.time()-start, tested, tested + len(neighbors), len(current.path())

    #If the lowest h is the current state, perform BFS to find a node with lower h
    if next_state == current:
      #Perform BFS
      BFSexpanded = 0
      next_state, BFSexpanded = aux_BFS(next_state, problem)
      expanded += BFSexpanded
    if next_state != None:
      current = next_state
      neighbors = current.expand(problem)
    expanded += 1
  return current, time.time()-start, tested, expanded, len(current.path()), time.time() - start, BFStested, BFSexpanded
