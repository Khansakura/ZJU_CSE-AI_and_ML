# Author:Khan
# -*- codeing = utf-8 -*-
# @Time : 2022/12/3 22:17
# @File : draw.py
# @Software: PyCharm
from tkinter import *
canvas= Canvas(width=800, height=600, bg='grey')

canvas.pack(expand=YES, fill=BOTH)

k= 1
j= 1
for i in range(0,26):
    canvas.create_oval(310 - k,250 - k,310 + k,250 + k, width=1)
    k+=j
mainloop()