# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
#Yuzhang Guo

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodWeight = 0
        if len(currentFood.asList()) > len(newFood.asList()):
            foodWeight = 10
        elif len(newFood.asList())>0:
            foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
            minDisToF = min(foodDistances)
            foodWeight = 10/minDisToF

        capsuleWeight = 0
        if len(currentCapsules) > len(newCapsules):
            capsuleWeight = 20
        elif len(newCapsules)>0:
            capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
            minDisToC = min(capsuleDistances)
            capsuleWeight = 20/minDisToC

        ghostWeight = 0
        ghostDistances = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates]
        if ghostDistances:
            minDisToG = min(ghostDistances)
            index = 0
            while minDisToG != ghostDistances[index]:
                index+=1
            scaredTime = newScaredTimes[index]
            if scaredTime > minDisToG:
                ghostWeight = 30
            elif minDisToG < 2:
                ghostWeight = -1000


        return foodWeight + capsuleWeight + ghostWeight + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()

        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth ==self.depth:
                return self.evaluationFunction(state)

            v = float("-inf")
            for action in state.getLegalActions():
                successorState = state.generateSuccessor(0,action)
                v = max(v,min_value(successorState, 1, depth))

            return v


        def min_value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex,action)
                if agentIndex+1 == numOfAgents:
                    v = min(v, max_value(successorState,depth+1))
                else:
                    v = min(v, min_value(successorState, agentIndex+1, depth))

            return v

        finalAction = None
        v = float("-inf")
        for action in gameState.getLegalActions():
            successorState = gameState.generateSuccessor(0,action)
            value = min_value(successorState, 1, 0)
            if value > v:
                v = value
                finalAction = action

        return finalAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()

        def max_value(state, depth, a, b):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = float("-inf")
            for action in state.getLegalActions():
                successorState = state.generateSuccessor(0, action)
                v = max(v, min_value(successorState, 1, depth, a, b))
                if v > b: return v
                a = max(a,v)

            return v

        def min_value(state, agentIndex, depth, a, b):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex + 1 == numOfAgents:
                    v = min(v, max_value(successorState, depth + 1, a, b))
                else:
                    v = min(v, min_value(successorState, agentIndex + 1, depth, a, b))

                if v < a: return v
                b = min(b, v)

            return v

        finalAction = None
        v = float("-inf")
        a = float("-inf")
        b = float("inf")
        for action in gameState.getLegalActions():
            successorState = gameState.generateSuccessor(0, action)
            value = min_value(successorState, 1, 0, a, b)
            if value > v:
                v = value
                finalAction = action
            if v > b: return v
            a = max(a,v)

        return finalAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()

        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = float("-inf")
            for action in state.getLegalActions():
                successorState = state.generateSuccessor(0, action)
                v = max(v, exp_value(successorState, 1, depth))

            return v

        def exp_value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = 0
            for action in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex + 1 == numOfAgents:
                    v += max_value(successorState, depth + 1)
                else:
                    v += exp_value(successorState, agentIndex + 1, depth)

            return v/float(len(state.getLegalActions(agentIndex)))

        finalAction = None
        v = float("-inf")
        for action in gameState.getLegalActions():
            successorState = gameState.generateSuccessor(0, action)
            value = exp_value(successorState, 1, 0)
            if value > v:
                v = value
                finalAction = action

        return finalAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
                    foodWeight is reciprical of average distance to the food left, encouraging pacman to get closer to food
                    10000/len(food) encourages pacman to eat food, especially when there is less food
                    500/len(capsule) encourage pacman to eat capsule
                    ghost weight keep pacman away or closer to ghost according to whether nearest ghost is scared or not
                    trapped weight is very high when all ghost are within 2 steps of pacman
                    getScore() is to punish pacman from stopping
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    ghostWeight = 0
    ghostDistances = [manhattanDistance(ghostState.getPosition(), pacPos) for ghostState in ghostStates]
    if ghostDistances:
        minDisToG = min(ghostDistances)
        index = 0
        while minDisToG != ghostDistances[index]:
            index += 1
        scaredTime = scaredTimes[index]
        if scaredTime > minDisToG:
            ghostWeight = 500/minDisToG
        elif minDisToG < 2:
            ghostWeight = -100000

    trapped = True
    for d in ghostDistances:
        if d >= 2:
            trapped = False
            break

    trappedWeight = 0
    if trapped:
        trappedWeight = -10000

    distance = 0
    for f in food:
        distance += manhattanDistance(pacPos,f)

    avg = distance/(len(food)+1)
    foodWeight = 500/(avg+1)

    return foodWeight + 10000/(len(food)+1) + 500/(len(capsules)+1) + ghostWeight + trappedWeight + 3*currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        numOfAgents = gameState.getNumAgents()
        self.depth = 3

        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return better(state)

            v = float("-inf")
            for action in state.getLegalActions():
                successorState = state.generateSuccessor(0, action)
                v = max(v, exp_value(successorState, 1, depth))

            return v

        def exp_value(state, agentIndex, depth):
            if state.isWin() or state.isLose():
                return better(state)

            pacPos = state.getPacmanPosition()
            selfPos = state.getGhostPosition(agentIndex)
            scared = (state.getGhostState(agentIndex).scaredTimer > 0)
            possibleActions = []
            directions = [Directions.EAST, Directions.SOUTH, Directions.WEST, Directions.NORTH]

            adjust = 0
            if scared:
                adjust = 2

            if selfPos[0] < pacPos[0]:
                possibleActions.append(directions[(0+adjust)%4])
            elif selfPos[0] > pacPos[0]:
                possibleActions.append(directions[(2+adjust)%4])

            if selfPos[1] < pacPos[1]:
                possibleActions.append(directions[(1+adjust)%4])
            elif selfPos[1] > pacPos[1]:
                possibleActions.append(directions[(3+adjust)%4])

            legalActions = state.getLegalActions(agentIndex)
            possibleActions = [action for action in possibleActions if action in legalActions]

            if len(possibleActions)>0:
                legalActions = possibleActions

            v = 0
            for action in legalActions:
                successorState = state.generateSuccessor(agentIndex, action)
                if agentIndex + 1 == numOfAgents:
                    v += max_value(successorState, depth + 1)
                else:
                    v += exp_value(successorState, agentIndex + 1, depth)

            return v/float(len(legalActions))

        finalAction = None
        v = float("-inf")
        for action in gameState.getLegalActions():
            successorState = gameState.generateSuccessor(0, action)
            value = exp_value(successorState, 1, 0)
            if value > v:
                v = value
                finalAction = action

        return finalAction