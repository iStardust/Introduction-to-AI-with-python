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
from math import sqrt, log
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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

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
      Your minimax agent (question 1)
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
        def value(gameState,searchdepth,agentIndex):
            agentNum=gameState.getNumAgents()
            #如果达到了搜索的深度,或者到达了游戏结束，那么就返回估值
            if searchdepth==self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            #否则的话，就开始进行minimax搜索
            
            #如果是吃豆人的话，那么吃豆人要选择子节点中value最大的
            if agentIndex==0:
                v=-999999
                legalActions=gameState.getLegalActions(0)
                for action in legalActions:
                    nextgameState=gameState.generateSuccessor(0,action)
                    v=max(v,value(nextgameState,searchdepth,(agentIndex+1)%agentNum))
                return v
            #否则选择子节点中value最小的
            else:
                #如果是最后一个agent搜索的话，那么层数+1
                if agentIndex==agentNum-1:
                    searchdepth+=1
                
                v=999999
                legalActions=gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    nextgameState=gameState.generateSuccessor(agentIndex,action)
                    v=min(v,value(nextgameState,searchdepth,(agentIndex+1)%agentNum))
                return v
        
        legalActions=gameState.getLegalActions(0)
        #valueDict以action作为键,以action所产生的子节点的value作为值
        valueDict=dict()
        for action in legalActions:
            nextgameState=gameState.generateSuccessor(0,action)
            valueDict[action]=value(nextgameState,0,1)
        bestvalue=-999999
        bestActions=[]
        #求出子节点value的最大值
        for actionvalue in valueDict.values():
            if actionvalue>=bestvalue:
                bestvalue=actionvalue
        #在所有子节点value的最大值中随机选一个
        for action,actionvalue in valueDict.items():
            if actionvalue==bestvalue:
                bestActions.append(action)
        chosenIndex=random.randrange(0,len(bestActions))
        return bestActions[chosenIndex]

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gameState,searchdepth,agentIndex,alpha,beta):
            agentNum=gameState.getNumAgents()
            #如果达到了搜索的深度,或者到达了游戏结束，那么就返回估值
            if searchdepth==self.depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)
            #否则的话，就开始进行alpha-beta剪枝搜索
            
            #如果是吃豆人的话，那么吃豆人要选择子节点中value最大的
            if agentIndex==0:
                v=-999999
                legalActions=gameState.getLegalActions(0)
                for action in legalActions:
                    nextgameState=gameState.generateSuccessor(0,action)
                    v=max(v,value(nextgameState,searchdepth,(agentIndex+1)%agentNum,alpha,beta))
                    #剪枝
                    if v>beta:
                        return v
                    alpha=max(alpha,v)
                return v
            #否则选择子节点中value最小的
            else:
                #如果是最后一个agent搜索的话，那么层数+1
                if agentIndex==agentNum-1:
                    searchdepth+=1
                v=999999
                legalActions=gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    nextgameState=gameState.generateSuccessor(agentIndex,action)
                    v=min(v,value(nextgameState,searchdepth,(agentIndex+1)%agentNum,alpha,beta))
                    #剪枝
                    if v<alpha:
                        return v
                    beta=min(beta,v)
                return v
        
        #valueDict以action作为键,以action所产生的子节点的value作为值
        valueDict=dict()
        alpha,beta=-999999,999999
        v=-999999
        legalActions=gameState.getLegalActions(0)
        for action in legalActions:
            nextgameState=gameState.generateSuccessor(0,action)
            actionvalue=value(nextgameState,0,1,alpha,beta)
            v=max(v,actionvalue)
            valueDict[action]=actionvalue
            #剪枝
            if v>beta:
                bestvalue=v
                break
            alpha=max(alpha,v)
        bestvalue=v
        bestActions=[]
        #在所有子节点value的最大值中随机选一个
        for action,actionvalue in valueDict.items():
            if actionvalue==bestvalue:
                bestActions.append(action)
        chosenIndex=random.randrange(0,len(bestActions))
        return bestActions[chosenIndex]
        #util.raiseNotDefined()

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """
    def __init__(self):
        self.depth=4
    
    def getAction(self, gameState):
        exploredPosSet=set()


        class Node:
            '''
            We have provided node structure that you might need in MCTS tree.
            '''
            def __init__(self, data):
                self.north = None
                self.east = None
                self.west = None
                self.south = None
                self.stop = None
                
                self.expandedList = [] # 里面的元素是元组(action,childnode)
                self.parent = None
                self.state = data[0]
                self.valueSum = data[1] 
                self.Nt = data[2] 
                self.agentIndex = data[3]
                #self.方向 = 朝那个方向采取action之后得到的Node
        data = [gameState, 0, 0 ,0]
        beginNode = Node(data) 
        C=sqrt(2)
        AGENTNUM=gameState.getNumAgents()
        def Selection(node):
            "*** YOUR CODE HERE ***"
            exploredPosSet.add(node.state.getPacmanPosition())
            #如果是叶节点就直接返回
            if node.expandedList==[]:
                return node
            #否则选择子节点中ucb值最大的
            maxucb = -999999
            chosenNode = node
            for expanded in node.expandedList:
                childnode=expanded[1]
                childucb=999999
                if childnode.Nt!=0:
                    childucb=(childnode.valueSum/childnode.Nt)+C*sqrt(log(node.Nt)/childnode.Nt)
                if childucb>maxucb:
                    maxucb=childucb
            selectList=[]
            for expanded in node.expandedList:
                childnode=expanded[1]
                childucb=999999
                if childnode.Nt!=0:
                    childucb=(childnode.valueSum/childnode.Nt)+C*sqrt(log(node.Nt)/childnode.Nt)
                if childucb==maxucb:
                    selectList.append(childnode)
            chosenIndex=random.randrange(0,len(selectList))
            return Selection(selectList[chosenIndex]) 
                        
            #util.raiseNotDefined()

        def Expansion(node):
            "*** YOUR CODE HERE ***"
            # 在子动作中随机选择一个扩展
            currentGameState=node.state
            actionList=currentGameState.getLegalActions(node.agentIndex)
            if actionList==[]:
                return node
            for action in actionList:
                nextAgentIndex=(node.agentIndex+1)%AGENTNUM
                nextState=currentGameState.generateSuccessor(node.agentIndex,action)
                if node.agentIndex!=0 or nextState.getPacmanPosition() not in exploredPosSet:
                    nextdata=[nextState,0,0,nextAgentIndex]
                    nextNode=Node(nextdata)
                    node.expandedList.append((action,nextNode))
                    nextNode.parent=node
                    exploredPosSet.add(nextState.getPacmanPosition())
            if node.expandedList==[]:
                return node
            randomChosenIndex=random.randrange(0,len(node.expandedList))
            return node.expandedList[randomChosenIndex][1]
            #util.raiseNotDefined()

        def Simulation(node):
            "*** YOUR CODE HERE ***"
            
            currentGameState=node.state
            agentIndex=node.agentIndex
            depth=0
            while True:
                actionList=currentGameState.getLegalActions(agentIndex)
                if agentIndex==AGENTNUM-1:
                    depth+=1
                if actionList==[] or depth==self.depth:
                    return HeuristicFunction(currentGameState)
                randomChosenIndex=random.randrange(0,len(actionList))
                chosenAction=actionList[randomChosenIndex]
                currentGameState=currentGameState.generateSuccessor(agentIndex,chosenAction)
                ghostStates=currentGameState.getGhostStates()
                for index in range(len(ghostStates)):
                    if currentGameState.data._eaten[index]:
                        return HeuristicFunction(currentGameState)
                agentIndex=(agentIndex+1)%AGENTNUM
            #util.raiseNotDefined()

        def Backpropagation(node):
            "*** YOUR CODE HERE ***"
            value=Simulation(node)
            node.Nt+=1
            if node.agentIndex==0:
                node.valueSum-=value
            else:
                node.valueSum+=value
            while node.parent!=None:
                node=node.parent
                node.Nt+=1
                if node.agentIndex==0:
                    node.valueSum-=value
                else:
                    node.valueSum+=value
            return
            #util.raiseNotDefined()
        WHOLEFOODNUM=gameState.getNumFood()
        WHOLECAPSULENUM=len(gameState.getCapsules())
        def HeuristicFunction(currentGameState):
            "*** YOUR CODE HERE ***"
            if currentGameState.isWin():
                return 100
            if currentGameState.isLose():
                return -100
            #设计的思路
            #1.如果鬼没有被惊吓，那么离得越远越好；如果鬼被惊吓了，那么离得越近越好
            #2.capsule剩余的数量越少越好，离得越近越好（类似p1的cornerProblem）
            #3.food的数量越少越好
            # newFood = successorGameState.getFood().asList()
            # newGhostStates = successorGameState.getGhostStates()
            # newScaredTimes = [
            # ghostState.scaredTimer for ghostState in newGhostStates]
            
            #离鬼越远（鬼受惊则反之），离食物越近，食物越少的情况下估值越高
            mapWidth=currentGameState.data.layout.width
            mapHeight=currentGameState.data.layout.height
            wholeMapDis=mapWidth+mapHeight
            pacmanPos=currentGameState.getPacmanPosition()
            foodPosList=currentGameState.getFood().asList()
            foodDisSum=0
            foodMinDis=wholeMapDis
            for foodPos in foodPosList:
                foodDisSum+=manhattanDistance(pacmanPos,foodPos)
                foodMinDis=min(foodMinDis,manhattanDistance(pacmanPos,foodPos))
            foodMinDis=foodMinDis/wholeMapDis
            foodAvgDis=(foodDisSum/currentGameState.getNumFood())/wholeMapDis#到食物的平均距离占总地图的比例
            FoodNum=currentGameState.getNumFood()
            foodRate=1
            if WHOLEFOODNUM!=0:
                foodRate=FoodNum/WHOLEFOODNUM#食物比例
            ghostStates=currentGameState.getGhostStates()
            ifscared=False
            newscarednum=0
            minGhostdis=1#离得最近的鬼的距离占地图最大曼哈顿距离的比值最小值
            for index,ghost in enumerate(ghostStates):
                disRate=manhattanDistance(pacmanPos,ghost.getPosition())/wholeMapDis
                if currentGameState.data._eaten[index]:
                    return 10000
                if disRate<minGhostdis:
                    minGhostdis=disRate
                    ifscared=False
                    if ghost.scaredTimer>0:
                        newscarednum+=1
                        ifscared=True         
            capsulePosList=currentGameState.getCapsules()
            capsuleNum=len(capsulePosList)
            capsuleRate=1
            if WHOLECAPSULENUM!=0:
                capsuleRate=capsuleNum/WHOLECAPSULENUM
            minCapsuleDis=1
            for capsulePos in capsulePosList:
                capsuleDis=manhattanDistance(pacmanPos,capsulePos)/wholeMapDis
                minCapsuleDis=min(minCapsuleDis,capsuleDis)
            ghostCloseEnough=0
            if minGhostdis<=0:
                minGhostdis=0.00001
            if minGhostdis*wholeMapDis<=2:
                ghostCloseEnough=1
            if foodRate<=0:
                foodRate=0.00001
            if capsuleRate<=0:
                capsuleRate=0.00001
            if minCapsuleDis<=0:
                minCapsuleDis=0.00001
            
                
            scaredflag=0.1
            if ifscared:
                scaredflag=-1000
            return ghostCloseEnough*scaredflag*0.01*log(minGhostdis)-100*log(foodRate)-1*pow(foodMinDis,2)-0.1*foodAvgDis-200*log(capsuleRate)-log(minCapsuleDis)

            util.raiseNotDefined()

        "*** YOUR CODE HERE ***"
        import time
        begintime=time.time()
        if gameState.getNumFood()/(gameState.data.layout.width*gameState.data.layout.height)<0.2:
            self.depth=5
        if gameState.getNumFood()/(gameState.data.layout.width*gameState.data.layout.height)<0.1:
            self.depth=6
        while True:
            if time.time()-begintime>0.2:
                break
            selectedNode=Selection(beginNode)
            chosenNode=selectedNode
            if selectedNode.Nt!=0:
                chosenNode=Expansion(selectedNode)
            Backpropagation(chosenNode)
        maxucb = -999999
        chosenAction = None
        for expanded in beginNode.expandedList:
            childnode=expanded[1]
            childucb=999999
            if childnode.Nt!=0:
                childucb=(childnode.valueSum/childnode.Nt)+C*sqrt(log(beginNode.Nt)/childnode.Nt)
            if childucb>maxucb:
                maxucb,chosenAction=childucb,expanded[0]
        return chosenAction
        util.raiseNotDefined()
