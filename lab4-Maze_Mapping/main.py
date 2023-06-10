# 导入相关包 
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt
import random
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

""" 创建迷宫并展示 """
maze = Maze(maze_size=10) # 随机生成迷宫
print(maze)
rewards = [] # 记录每走一步的奖励值
actions = [] # 记录每走一步的移动方向

# 循环、随机移动机器人10次，记录下奖励
for i in range(10):
    valid_actions = maze.can_move_actions(maze.sense_robot())
    action = random.choice(valid_actions)
    rewards.append(maze.move_robot(action))
    actions.append(action)

print("the history of rewards:", rewards)
print("the actions", actions)

# 输出机器人最后的位置
print("the end position of robot:", maze.sense_robot())

# 打印迷宫，观察机器人位置
print(maze)

def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []
    
    # -----------------请实现你的算法代码--------------------------------------
    #参考广度优先算法，模型采用深度优先算法进行机器人走迷宫问题的求解

    # 机器人移动方向
    move_map = {
        'u': (-1, 0), # up
        'r': (0, +1), # right
        'd': (+1, 0), # down
        'l': (0, -1), # left
    }
    class SearchTree(object):


        def __init__(self, loc=(), action='', parent=None):
            """
            初始化搜索树节点对象
            :param loc: 新节点的机器人所处位置
            :param action: 新节点的对应的移动方向
            :param parent: 新节点的父辈节点
            """

            self.loc = loc  # 当前节点位置
            self.to_this_action = action  # 到达当前节点的动作
            self.parent = parent  # 当前节点的父节点
            self.children = []  # 当前节点的子节点

        def add_child(self, child):
            """
            添加子节点
            :param child:待添加的子节点
            """
            self.children.append(child)

        def is_leaf(self):
            """
            判断当前节点是否是叶子节点
            """
            return len(self.children) == 0
    def expand(maze, is_visit_m, node):
        """
        拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
        :param maze: 迷宫对象
        :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
        :param node: 待拓展的叶子节点
        """
        child_number = 0  # 记录叶子节点个数
        can_move = maze.can_move_actions(node.loc)
        for a in can_move:
            new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
            if not is_visit_m[new_loc]:
                child = SearchTree(loc=new_loc, action=a, parent=node)
                node.add_child(child)
                child_number+=1
        return child_number  # 返回叶子节点个数
    
    def back_propagation(node):
        """
        回溯并记录节点路径
        :param node: 待回溯节点
        :return: 回溯路径
        """
        path = []
        while node.parent is not None:
            path.insert(0, node.to_this_action)
            node = node.parent
        return path
       
        
    def DFS(maze):
        """
        深度搜索算法的实现：
        """
        #准备过程
        start = maze.sense_robot()#机器人初始位置获取
        root = SearchTree(loc=start)#以初始位置建立待搜索对象
        queue=[root]#堆栈建立
        height,width, _=maze.maze_data.shape#获取迷宫对象的结点深度信息
        is_visit_m=np.zeros((height,width),dtype=np.int)#储存已经搜索过的结点
        path=[]#路径保存
        ans=0#堆栈指针变量
        #搜索过程
        while True:
            current_node=queue[ans]
            if current_node.loc == maze.destination:#目标判断
                path=back_propagation(current_node)#回溯并记录节点路径
                break
            if current_node.is_leaf() and is_visit_m[current_node.loc]==0:#未搜索过的叶结点
                is_visit_m[current_node.loc]=1#标记结点已经被搜索
                child_number=expand(maze,is_visit_m,current_node)#扩展子节点
                ans+=child_number#准备
                for child in current_node.children:
                    queue.append(child)
            else:
                queue.pop(ans)#无路则出栈
                ans-=1
            #出队列
            
        return path
    
    path=DFS(maze)
    
    # -----------------------------------------------------------------------
    return path
  
  from QRobot import QRobot
from Maze import Maze
from Runner import Runner

"""  Qlearning 算法相关参数： """

epoch = 10  # 训练轮数
epsilon0 = 0.5  # 初始探索概率
alpha = 0.5  # 公式中的 ⍺
gamma = 0.9  # 公式中的 γ
maze_size = 5  # 迷宫size

""" 使用 QLearning 算法训练过程 """

g = Maze(maze_size=maze_size)
r = QRobot(g, alpha=alpha, epsilon0=epsilon0, gamma=gamma)

runner = Runner(r)
runner.run_training(epoch, training_per_epoch=int(maze_size * maze_size * 1.5))
runner.plot_results() # 输出训练结果，可根据该结果对您的机器人进行分析。

class Robot(TorchRobot):
    
    def __init__(self,maze):
        super(Robot,self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 10.0,
            "destination": -maze.maze_size ** 2.0,
            "default": 1.0,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        self.loss_list = self.train()
    def train(self):
        loss_list = []
        batch_size = len(self.memory)

        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            success = False
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    return loss_list

    def train_update(self):
        def state_train():
            state=self.sense_state()
            return state
        def action_train(state):
            action=self._choose_action(state)
            return action
        def reward_train(action):
            reward=self.maze.move_robot(action)
            return reward
        state = state_train()
        action = action_train(state)
        reward = reward_train(action)
        return action, reward

    def test_update(self):
        def state_test():
            state = torch.from_numpy(np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
            return state
        state = state_test()
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()
        def action_test(q_value):
            action=self.valid_action[np.argmin(q_value).item()]
            return action
        def reward_test(action):
            reward=self.maze.move_robot(action)
            return reward
        action = action_test(q_value)
        reward = reward_test(action)
        return action, reward
 class Robot(QRobot):
    
    valid_action=['u','d','l','r']
    

    def __init__(self,maze,alpha=0.5,gamma=0.9,epsilon=0.5):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        
        self.maze = maze
        self.state=None
        self.action=None
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon = epsilon
        
        self.maze.reset_robot()
        self.state=self.maze.sense_robot()
        
        if self.state not in self.q_table:
            self.q_table[self.table]={a: 0.0 for a in self.vaild_action}
        
    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """


        # -----------------请实现你的算法代码--------------------------------------
        self.state=self.maze.sense_robot()#获取当前位置为输入位置
        #查看Q表中是否有该状态，没有则加入
        if self.state not in self.q_table:
            self.q_table[self.state]={a: 0.0 for a in self.valid_action} #创建字典
        
        # 采用蒙特卡洛方法对action为机器人选择的动作
        action = random.choice(self.valid_action) if random.random() < self.epsilon else max(self.q_table[self.state], key=self.q_table[self.state].get)
        reward = self.maze.move_robot(action)  #返回的奖励值
        next_state = self.maze.sense_robot()  # 动作后新位置
        
        # 检索Q表，如果当前的新状态不存在则添加进入Q表
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.valid_action} #创建字典
         
        # 更新Q值表
        current_r = self.q_table[self.state][action]
        update_r = reward + self.gamma * float(max(self.q_table[next_state].values()))
        self.q_table[self.state][action] = (1 - self.alpha) * (update_r - current_r) + self.alpha * self.q_table[self.state][action]       
        self.epsilon *= 0.5  # 衰减随机选择动作的可能性

        # -----------------------------------------------------------------------

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """


        # -----------------请实现你的算法代码--------------------------------------
        self.state=self.maze.sense_robot()
        
        if self.state not in self.q_table:
            self.q_table[self.state]={a: 0.0 for a in self.valid_action} #创建字典
        
        action = max(self.q_table[self.state],key=self.q_table[self.state].get)
        
        reward = self.maze.move_robot(action)
        
        # -----------------------------------------------------------------------

        return action, reward
