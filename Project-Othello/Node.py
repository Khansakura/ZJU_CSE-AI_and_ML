import datetime
import numpy as np
from board import Board
from copy import deepcopy
import math
import random
from player import RandomPlayer
from game import Game
from player import HumanPlayer



class Node:
    def __init__(self, board, parent=None, action=None, color=""):
        self.visits = 0     #访问次数
        self.reward = 0.0   #期望值
        self.board = board  #当前时刻棋盘状态
        self.children = []  #子节点数组
        self.parent = parent #父节点
        self.action = action #从父节点传递到本结点采取的动作
        self.color = color #结点的玩家阵营

    def add_child(self, child_board, action, color):
        '''
        为当前节点创造子节点
        para：当前节点对象self，child_board子节点状态，action子节点采取动作，color子节点玩家阵营
        '''
        child_node=Node(child_board, parent = self, action = action, color = color)
        self.children.append(child_node)

    def full_expand(self):
        '''
        判断结点是否完全展开
        '''
        action = list(self.board.get_legal_actions(self.color))
        if len(self.children) == len(action):
            return True
        return False

    
 





