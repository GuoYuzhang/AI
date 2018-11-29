# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
from functools import partial

import util




class Node:
    def __init__(self, state, parent=None, action=None,cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def getCost(self):
        if self.parent:
            return self.cost + self.parent.getCost()
        return self.cost

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # pass stack to the graph search algorithm
    return __generalSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    # pass queue to the graph search algorithm
    return __generalSearch(problem,util.Queue())

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # pass priority queue which ranks node based on distance from start position to the graph search algorithm
    return __generalSearch(problem, util.PriorityQueueWithFunction(Node.getCost))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    # return node's distance from start position + heuristic calculated by heuristic function
    def totalCost(n):
        return n.getCost()+heuristic(n.state, problem)

    # pass priority queue which ranks node bases on totalCost() to the graph search algorithm
    return __generalSearch(problem, util.PriorityQueueWithFunction(totalCost))

def __generalSearch(problem, fringe):
    closed_set , actions = set(), []
    fringe.push(Node(problem.getStartState()))
    while not fringe.isEmpty():
        current = fringe.pop()
        if problem.isGoalState(current.state):
            # if goal state is reached, keep adding actions from goal state back to start state to a list
            while current.parent:
                actions.append(current.action)
                current = current.parent
            # return the reversed list
            return actions[::-1]
        if current.state not in closed_set:
            closed_set.add(current.state)
        else: continue

        successors = problem.getSuccessors(current.state)
        for state, action, cost in successors:
            child = Node(state, parent=current, action=action,cost=cost)
            fringe.push(child)

    return None



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch