# !/usr/bin/Anaconda3/python
# -*- coding: utf-8 -*-

from goto import with_goto
import tkinter as tk
import numpy as np
import tkinter.messagebox as messagebox
from collections import Counter
from tkinter import StringVar, IntVar
from func_timeout import func_timeout, FunctionTimedOut
import datetime
from board import Board
from copy import deepcopy
from tkinter import font


class Game(object):
    def __init__(self, black_player, white_player):
        self.board = Board()  # 棋盘
        # 定义棋盘上当前下棋棋手，先默认是 None
        self.current_player = None
        self.black_player = black_player  # 黑棋一方
        self.white_player = white_player  # 白棋一方
        self.black_player.color = "X"
        self.white_player.color = "O"

    def switch_player(self, black_player, white_player):
        """
        游戏过程中切换玩家
        :param black_player: 黑棋
        :param white_player: 白棋
        :return: 当前玩家
        """
        # 如果当前玩家是 None 或者 白棋一方 white_player，则返回 黑棋一方 black_player;
        if self.current_player is None:
            return black_player
        else:
            # 如果当前玩家是黑棋一方 black_player 则返回 白棋一方 white_player
            if self.current_player == self.black_player:
                return white_player
            else:
                return black_player

    def print_winner(self, winner):
        """
        打印赢家
        :param winner: [0,1,2] 分别代表黑棋获胜、白棋获胜、平局3种可能。
        :return:
        """
        print(['黑棋获胜!', '白棋获胜!', '平局'][winner])

    def force_loss(self, is_timeout=False, is_board=False, is_legal=False):
        """
         落子3个不合符规则和超时则结束游戏,修改棋盘也是输
        :param is_timeout: 时间是否超时，默认不超时
        :param is_board: 是否修改棋盘
        :param is_legal: 落子是否合法
        :return: 赢家（0,1）,棋子差 0
        """

        if self.current_player == self.black_player:
            win_color = '白棋 - O'
            loss_color = '黑棋 - X'
            winner = 1
        else:
            win_color = '黑棋 - X'
            loss_color = '白棋 - O'
            winner = 0

        if is_timeout:
            print('\n{} 思考超过 60s, {} 胜'.format(loss_color, win_color))
        if is_legal:
            print('\n{} 落子 3 次不符合规则,故 {} 胜'.format(loss_color, win_color))
        if is_board:
            print('\n{} 擅自改动棋盘判输,故 {} 胜'.format(loss_color, win_color))

        diff = 0

        return winner, diff

    def run(self):
        """
        运行游戏
        :return:
        """
        # 定义统计双方下棋时间
        total_time = {"X": 0, "O": 0}
        # 定义双方每一步下棋时间
        step_time = {"X": 0, "O": 0}
        # 初始化胜负结果和棋子差
        winner = None
        diff = -1

        #从这里开始改的喔
        location = [-1, -1]
        canvas = ''
        board = ''

        def change_num(action):
            """
            棋盘坐标转化为数字坐标
            :param action:棋盘坐标，比如A1
            :return:数字坐标，比如 A1 --->(0,0)
            """
            row, col = str(action[1]).upper(), str(action[0]).upper()
            if row in '12345678' and col in 'ABCDEFGH':
                # 坐标正确
                x, y = '12345678'.index(row), 'ABCDEFGH'.index(col)
                return x, y

        def num_change(action):
            """
            数字坐标转化为棋盘坐标
            :param action:数字坐标 ,比如(0,0)
            :return:棋盘坐标，比如 （0,0）---> A1
            """
            row, col = action
            l = [0, 1, 2, 3, 4, 5, 6, 7]
            if col in l and row in l:
                return chr(ord('A') + col) + str(row + 1)

        def change_board(self):
            current_situation = np.zeros((9, 9))
            #color = "X" if self.current_player == self.black_player and self.current_player == None else "O"
            color = "X"
            legal_actions = list(self.board.get_legal_actions(color))
            action_list=[]
            for legal_action in legal_actions:
                x, y = change_num(legal_action)
                action_list.append((x,y))
            for i in range(8):
                # i 是行数，从0开始，j是列数，也是从0开始
                for j in range(8):
                    if self.board[i][j] == 'X':
                        current_situation[i][j] = -1
                    elif self.board[i][j] == 'O':
                        current_situation[i][j] = 1
            for index in action_list:
                     current_situation[index[0]][index[1]] = 2
            return current_situation

        def draw_Chess_from_Maxtrix(current_situation):
            for i in range(len(current_situation)):
                for j in range(len(current_situation[0])):
                    if current_situation[i][j] == 1:
                        canvas.create_oval(j * 50 + 6, i * 50 + 6, j * 50 + 44, i * 50 + 44, fill='white')
                    if current_situation[i][j] == -1:
                        canvas.create_oval(j * 50 + 6, i * 50 + 6, j * 50 + 44, i * 50 + 44, fill='black')
                    if current_situation[i][j] == 2:
                        canvas.create_oval(j * 50 + 6, i * 50 + 6, j * 50 + 44, i * 50 + 44, outline='red')
                    if current_situation[i][j] == 0:
                        canvas.create_oval(j * 50 + 6, i * 50 + 6, j * 50 + 44, i * 50 + 44, outline='cornsilk')
        # 游戏开始
        print('\n=====开始游戏!=====\n')
        # 棋盘初始化
        self.board.display(step_time, total_time)
        current = change_board(self)
        print(current)
        rt = tk.Tk(className="黑白棋")
        rt.resizable(0, 0)
        canvas = tk.Canvas(rt, width=700, height=500, bg='cornsilk')
        canvas.bind()
        canvas.pack(padx=10, pady=10)
        white_label = tk.Label(rt, width=6, height=1, text='电脑')
        white_label.place(x=600, y=200)
        black_label = tk.Label(rt, width=6, height=1, text='玩家')
        black_label.place(x=600, y=260)
        drop_label = tk.Label(rt, width=8, height=1, text='合法落子位置')
        drop_label.place(x=600, y=320)
        name_label = tk.Label(rt, width=50, height=3, text='黑白棋，游戏通过相互翻转对方的棋子，最后以棋盘上谁的棋子多来判断胜负。')
        name_label.place(x=100, y=450)
        label1 = tk.Label(rt, text="黑白棋大作业——周克涵、林骏杰、郭兴华").pack()




        def drop(event):
            row = int(event.y / 50)
            col = int(event.x / 50)
            global location
            location = [row, col]
            # 切换当前玩家,如果当前玩家是 None 或者白棋 white_player，则返回黑棋 black_player;
            #  否则返回 white_player。
            self.current_player = self.switch_player(self.black_player, self.white_player)
            start_time = datetime.datetime.now()
            # 当前玩家对棋盘进行思考后，得到落子位置
            # 判断当前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 获取当前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            print("%s合法落子坐标列表：" % color, legal_actions)
            if len(legal_actions) == 0:
                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束，双方都没有合法位置
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

            else:
                board = deepcopy(self.board._board)
                # legal_actions 不等于 0 则表示当前下棋方有合法落子位置
                i = 0
                try:
                    # 获取落子位置
                    # action = func_timeout(60, self.current_player.get_move,
                    #                      kwargs={'board': self.board})

                    action = num_change(location)
                    # if self.current_player == self.black_player:
                    #    action = func_timeout(60, self.current_player.get_move,
                    #                           kwargs={'board': self.board})

                    # 如果 action 是 Q 则说明人类想结束比赛
                    if action == "Q":
                        # 说明人类想结束游戏，即根据棋子个数定输赢。
                        pass
                    if action not in legal_actions:
                        # 判断当前下棋方落子是否符合合法落子,如果不合法,则需要对方重新输入
                        i = i + 1
                        tk.messagebox.showinfo('提醒', '你落子不符合规则,请重新落子！')
                        print("你落子不符合规则,请重新落子！")
                        return False
                    if i > 3:
                        # 落子3次不合法，结束游戏！
                        tk.messagebox.showinfo('提醒', '落子三次不合法，你失败了！')
                        print("落子三次不合法，你失败了！")
                        winner, diff = self.force_loss(is_legal=True)
                        if winner == 0:
                            messagebox.showinfo('提醒', '您胜利了！')
                        elif winner == 1:
                            messagebox.showinfo('提醒', '你失败了！')
                        else:
                            messagebox.showinfo('提醒', '平局！')

                except FunctionTimedOut:
                    # 落子超时，结束游戏
                    tk.messagebox.showinfo('提醒', '落子超时，你失败了！')
                    print("落子超时，你失败了！")
                    winner, diff = self.force_loss(is_timeout=True)
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

                # 结束时间
                end_time = datetime.datetime.now()

                if board != self.board._board:
                    # 修改棋盘，结束游戏！
                    winner, diff = self.force_loss(is_board=True)
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

                if  action  in legal_actions:
                    self.board._move(action, color)


                if action == "Q":
                    # 说明人类想结束游戏，即根据棋子个数定输赢。
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

                if action == 'W':
                    pass
                else:
                    # 统计一步所用的时间
                    es_time = (end_time - start_time).seconds
                    if es_time > 60:
                        # 该步超过60秒则结束比赛。
                        print('\n{} 思考超过 60s'.format(self.current_player))
                        winner, diff = self.force_loss(is_timeout=True)

                    if action != "W":
                        # 当前玩家颜色，更新棋局
                        self.board._move(action, color)
                    # 统计每种棋子下棋所用总时间
                    if self.current_player == self.black_player:
                        # 当前选手是黑棋一方
                        step_time["X"] = es_time
                        total_time["X"] += es_time
                    else:
                        step_time["O"] = es_time
                        total_time["O"] += es_time
                    # 显示当前棋盘

                    self.board.display(step_time, total_time)

                    # 判断游戏是否结束
                    if self.game_over():
                        # 游戏结束
                        winner, diff = self.board.get_winner()  # 得到赢家 0,1,2

            draw_Chess_from_Maxtrix(change_board(self))

            self.current_player = self.switch_player(self.black_player, self.white_player)
            start_time = datetime.datetime.now()
            # 当前玩家对棋盘进行思考后，得到落子位置
            # 判断当前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 获取当前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            print("%s合法落子坐标列表：" % color, legal_actions)
            if len(legal_actions) == 0:
                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束，双方都没有合法位置
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')
            else:

                board = deepcopy(self.board._board)
                # legal_actions 不等于 0 则表示当前下棋方有合法落子位置
                i = 0

                action = func_timeout(60, self.current_player.get_move,
                                      kwargs={'board': self.board})

                # 结束时间
                end_time = datetime.datetime.now()
                if board != self.board._board:
                    # 修改棋盘，结束游戏！
                    winner, diff = self.force_loss(is_board=True)
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

                draw_Chess_from_Maxtrix(change_board(self))

                # 统计一步所用的时间
                es_time = (end_time - start_time).seconds
                if es_time > 60:
                    # 该步超过60秒则结束比赛。
                    print('\n{} 思考超过 60s'.format(self.current_player))
                    winner, diff = self.force_loss(is_timeout=True)
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

                if action != "W":
                    # 当前玩家颜色，更新棋局
                    self.board._move(action, color)
                # 统计每种棋子下棋所用总时间

                # 显示当前棋盘

                self.board.display(step_time, total_time)

                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束
                    winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    if winner == 0:
                        messagebox.showinfo('提醒', '您胜利了！')
                    elif winner == 1:
                        messagebox.showinfo('提醒', '你失败了！')
                    else:
                        messagebox.showinfo('提醒', '平局！')

            draw_Chess_from_Maxtrix(change_board(self))





        canvas.create_line(0, 0, 0, 400, fill='black', width=10)
        canvas.create_line(0, 0, 400, 0, fill='black', width=10)
        canvas.create_line(400, 400, 0, 400, fill='black', width=3)
        canvas.create_line(400, 400, 400, 0, fill='black', width=3)
        for i in range(8):
            canvas.create_line(50 * i, 0, 50 * i, 400, fill='black', width=2)
            canvas.create_line(0, 50 * i, 400, 50 * i, fill='black', width=2)
        canvas.bind(sequence='<Button-1>', func=drop)
        canvas.create_oval(540, 180, 580, 220, fill='white')
        canvas.create_oval(540, 240, 580, 280, fill='black')
        canvas.create_oval(540, 300, 580, 340, outline='red')
        draw_Chess_from_Maxtrix(change_board(self))


        rt.mainloop()


        print('\n=====游戏结束!=====\n')
        self.board.display(step_time, total_time)
        self.print_winner(winner)

        # 返回'black_win','white_win','draw',棋子数差
        if winner is not None and diff > -1:
            result = {0: 'black_win', 1: 'white_win', 2: 'draw'}[winner]

            # return result,diff

    def game_over(self):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """

        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(self.board.get_legal_actions('X'))
        w_list = list(self.board.get_legal_actions('O'))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over

#
#
# if __name__ == '__main__':
#     from Human_player import HumanPlayer
#     from Random_player import RandomPlayer
#     from AIPlayer import AIPlayer
#
#     # x = HumanPlayer("X")
#     x = RandomPlayer("X")
#     o = RandomPlayer("O")
# #     # x = AIPlayer("X")
# #     o = AIPlayer("O")
#     game = Game(x, o)
#     game.run()
