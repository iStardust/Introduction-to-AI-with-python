# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #易小鱼 2023.03.10
    #search steps
    #0.begin with a frontier that contains the initial_state
    #loop
    #1.check if the frontier is empty,if so, break
    #2.pop out a state from the frontier
    #3.check if the popped-out state is the goal_state,if so, return the solution;
    #add the goal-checked state into the visited set
    #4.if not so,add all the neighbor states into the frontier

    initial_state=problem.getStartState()
    actionsLst=[]#用来记录从开始到达当前的节点的动作列表
    stackfrontier=util.Stack()
    stackfrontier.push((initial_state,actionsLst))
    visited=[]
    while not stackfrontier.isEmpty():
        state,actions=stackfrontier.pop()
        if problem.isGoalState(state):
            return actions
        visited.append(state)
        for next in problem.getSuccessors(state):
            next_state=next[0]
            next_action=next[1]
            if next_state not in visited:
                stackfrontier.push((next_state,actions+[next_action]))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
     """Search the shallowest nodes in the search tree first."""
     #python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs --frameTime 0
     #python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5 --frameTime 0
     "*** YOUR CODE HERE ***"
     Frontier = util.Queue()
     Visited = []
     Frontier.push( (problem.getStartState(), []) )
     #print 'Start',problem.getStartState()
     Visited.append( problem.getStartState() )

     while Frontier.isEmpty() == 0:
         state, actions = Frontier.pop()
         if problem.isGoalState(state):
             #print 'Find Goal'
             return actions 
         for next in problem.getSuccessors(state):
             n_state = next[0]
             n_direction = next[1]
             if n_state not in Visited:
                 Frontier.push( (n_state, actions + [n_direction]) )
                 Visited.append( n_state )

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #易小鱼 2023.03.10
    #steps
    #0.begin with a frontier that contains initial node
    #1.check if the frontier is empty
    #2.pop out the least total cost node from the frontier
    #3.goal test, if yes, return actions
    #4.add all neighbor nodes into the frontier, in the order of cost function, if existed,check whether replacable
    initialState=problem.getStartState()
    actionsLst=[]
    initialCost=0
    PriorityFrontier=util.PriorityQueue()
    PriorityFrontier.push((initialState,actionsLst),initialCost)
    visited=[]
    while not PriorityFrontier.isEmpty():
        state,action=PriorityFrontier.pop()
        if problem.isGoalState(state):
            return action
        visited.append(state)
        for next in problem.getSuccessors(state):
            n_state=next[0]
            n_direction=next[1]
            n_actions=action+[n_direction]
            n_totalcost=problem.getCostOfActions(n_actions)
            if n_state not in visited:
                '''
                ！！！！注意！！！！
                在util.py中，在进行PriorityQueue的Update时，是如以下的步骤：
                1.判断当前打算添加进去的item是否已经存在在队列中，若已经存在，
                则将队列中该item的优先值与打算添加进去的item的优先值进行比较，
                取其小者入队列，剩下的删除
                2.若不存在在队列中，则直接添加
                然而，这样的逻辑存在一个不符合我的需求的地方就是，在判断item是否存在在队列中时，
                util.py中使用的是"=="进行判断，但是我在这个函数中队列中的item是包含了
                (state,action)两个部分，即包含了当前所处的位置和从起点抵达该位置的路径两部分；
                也就是说，我们想要达到的效果是只要state相同，就认为该个item已经存在于队列中，
                但是util.py中的PriorityQueue的update函数会认为要state和action都相同才认为item已存在，
                与我的目的不符合，于是我参照util.py，自己写了一个对PriorityQueue进行更新的代码如下
                '''
                for index, (p, c, i) in enumerate(PriorityFrontier.heap):
                    if i[0] == n_state:#只判断当前所处state是否相同
                        if p <= n_totalcost:
                            break
                        del PriorityFrontier.heap[index]
                        PriorityFrontier.heap.append((n_totalcost, c, (n_state,n_actions)))
                        heapq.heapify(PriorityFrontier.heap)
                        break
                else:
                    PriorityFrontier.push((n_state,n_actions), n_totalcost)
    

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #aStar与UCS感觉类似，区别就是aStar中的优先值是成本+启发两部分组成
    initialState=problem.getStartState()
    actionsLst=[]
    initialPriority=0+heuristic(initialState,problem)
    PriorityFrontier=util.PriorityQueue()
    PriorityFrontier.push((initialState,actionsLst),initialPriority)
    visited=[]
    while not PriorityFrontier.isEmpty():
        state,action=PriorityFrontier.pop()
        if problem.isGoalState(state):
            return action
        visited.append(state)
        for next in problem.getSuccessors(state):
            n_state=next[0]
            n_direction=next[1]
            n_actions=action+[n_direction]
            n_totalcost=problem.getCostOfActions(n_actions)
            n_heuristic=heuristic(n_state,problem)
            n_priority=n_totalcost+n_heuristic
            if n_state not in visited:
                for index, (p, c, i) in enumerate(PriorityFrontier.heap):
                    if i[0] == n_state:#只判断当前所处state是否相同
                        if p <= n_priority:
                            break
                        del PriorityFrontier.heap[index]
                        PriorityFrontier.heap.append((n_priority, c, (n_state,n_actions)))
                        heapq.heapify(PriorityFrontier.heap)
                        break
                else:
                    PriorityFrontier.push((n_state,n_actions), n_priority)
    

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
