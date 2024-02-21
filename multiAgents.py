# multiAgents.py
# --------------
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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

    def Min_Value(self, gamestate, agentindex, current_depth):
        legal_actions = gamestate.getLegalActions(agentindex)

        value = float('inf')  # initialization
        move = 'Stop'

        if (gamestate.isWin() == False) and (gamestate.isLose() == False):

            for action in legal_actions:

                if agentindex < gamestate.getNumAgents() - 1:  # if the ghost is not the last it calls another ghost
                    (value2, move2) = self.Min_Value(gamestate.generateSuccessor(agentindex, action), agentindex + 1,
                                                     current_depth)

                elif current_depth < self.depth:  # if the ghost is the last but the maximum depth hasn't been reached it calls pacman again increasing the current depth
                    current_depth += 1
                    (value2, move2) = self.Max_Value(gamestate.generateSuccessor(agentindex, action), current_depth)
                    current_depth -= 1  # When Pacman is done it returns the current depth back to normal

                else:  # if it is the last and the maximum depth has been reached it examines the score of every legal action
                    (value2, move2) = (self.evaluationFunction(gamestate.generateSuccessor(agentindex, action)), action)

                if value2 < value:  # if the score of the action is lower than the current choice it gets subsituted
                    (value, move) = (value2, action)

        else:  # if the current gamestate is terminal it returns the value of the state
            value = self.evaluationFunction(gamestate)

        return (value, move)

    def Max_Value(self, gamestate, current_depth):
        legal_actions = gamestate.getLegalActions(0)

        value = float('-inf')  # initialization
        move = 'Stop'

        if gamestate.isWin() == False and gamestate.isLose() == False:

            for action in legal_actions:
                if current_depth <= self.depth:  # Pacman calls the first ghost to know the score of a certain action
                    (value2, move2) = self.Min_Value(gamestate.generateSuccessor(0, action), 1, current_depth)

                if value2 > value:  # if score of the action is greater than the current choice it substitutes the value and the move
                    (value, move) = (value2, action)

        else:  # if the current gamestate is terminal it returns the value of the state
            value = self.evaluationFunction(gamestate)

        return (value, move)

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        current_depth = 1
        (value, move) = self.Max_Value(gameState, current_depth)  # Code starts by calling Pacman

        return move

# python autograder.py -q q2

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # alpha - the value of the best choice we have found so far at the choice point along the path for max
        # beta - the value of the best choice we have found so far at the choice point along the path for min

        def minValue(gameState, depth, agent, alpha, beta):

            # Stores the action with the corresponding value (starts in infinity)
            minimum = ["", float("inf")]

            # Legal actions for the agent in the actual gameState
            legal_actions = gameState.getLegalActions(agent)

            # If there is no legal action probably is because it is going to lose
            if not legal_actions:
                return self.evaluationFunction(gameState)  # Call the evaluation function

            # Iterate through every possible position
            for action in legal_actions:

                current_gameState = gameState.generateSuccessor(agent, action)
                # ^^^ current gameState if X action is taken

                current = next_action(current_gameState, depth, agent + 1, alpha, beta)
                # ^^^ value for the next agent, due to X action

                # Gets the value of the next agent
                if type(current) is not list:  # if there is just one
                    newValue = current
                else:
                    newValue = current[1]

                # Compare to find if it is our new minimum value or not
                if newValue < minimum[1]:
                    minimum = [action, newValue]

                if newValue < alpha:
                    return [action, newValue]

                # Updates beta taking the small gotten value
                beta = min(beta, newValue)

            return minimum

        def maxValue(gameState, depth, agent, alpha, beta):

            # Stores the action with the corresponding value (starts in - infinity)
            maximum = ["", -float("inf")]

            # Legal actions for the agent in the actual gameState
            legal_actions = gameState.getLegalActions(agent)

            # If there is no legal action probably is because it is going to lose
            if not legal_actions:
                return self.evaluationFunction(gameState)  # Call the evaluation function

            # Iterate through every possible position again
            for action in legal_actions:

                current_gameState = gameState.generateSuccessor(agent, action)
                # ^^^ current gameState if X action is taken

                current = next_action(current_gameState, depth, agent + 1, alpha, beta)
                # ^^^ value for the next agent, due to X action

                # Gets the value of the next agent
                if type(current) is not list:
                    newValue = current
                else:
                    newValue = current[1]

                # Compare to find if it is our new maximum value or not
                if newValue > maximum[1]:
                    maximum = [action, newValue]
                if newValue > beta:
                    return [action, newValue]

                # Updates alpha taking the biggest gotten value
                alpha = max(alpha, newValue)

            return maximum

        def next_action(gameState, depth, agent, alpha, beta):

            # If agent is bigger than the number of agents in the current gameState => increase the depth
            if agent >= gameState.getNumAgents():
                depth += 1
                agent = 0

            # If you get to the final layer in the action tree or the game is lost or won (end of the game)
            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)  # Call the evaluation function

            # Depending on which agent we would want to maximize or minimize
            elif (agent == 0):
                return maxValue(gameState, depth, agent, alpha, beta)

            else:
                return minValue(gameState, depth, agent, alpha, beta)

        actionsList = next_action(gameState, 0, 0, -float("inf"), float("inf"))

        return actionsList[0]

# python autograder.py -q q3
# python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
