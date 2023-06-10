# 首先 import 一些主要的包
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
# 简单读出一个股票
# 获取文件名
file_name = 'train_data.npy'
# 读取数组
data = np.load(file_name)
# 简单展示信息
data
plt.figure(figsize=(20,10))

# 绘画该股票不同的时间段的图像
plt.plot(data,c='blue')

# 展示图像
plt.show()

diff = np.diff(data)
# 新建一个图像
plt.figure(figsize=(20,10))

# 绘画该股票不同的时间段的图像
plt.plot(diff,c='blue')

# 展示图像
plt.show()

import difflib
diff2 = np.diff(diff)
# 新建一个图像
plt.figure(figsize=(20,10))

# 绘画该股票不同的时间段的图像
plt.plot(diff2,c='blue')

# 展示图像
plt.show()

# 生成题目所需的训练集合
def generate_data(data):
    
    # 记录 data 的长度
    n = data.shape[0]
    
    # 目标是生成可直接用于训练和测试的 x 和 y
    x = []
    y = []
        
    # 建立 (14 -> 1) 的 x 和 y
    for i in range(15, n):
        x.append(data[i-15:i-1])
        y.append(data[i-1])
            
    # 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)
        
    return x,y

x,y = generate_data(data)
print('x.shape : ', x.shape)
print('y.shape : ', y.shape)

def generate_training_data(x, y):
    # 样本总数
    num_samples = x.shape[0]
    # 测试集大小
    num_test = round(num_samples * 0.2)
    # 训练集大小
    num_train = round(num_samples * 0.7)
    # 校验集大小
    num_val = num_samples - num_test - num_train
    
    # 训练集拥有从 0 起长度为 num_train 的样本
    x_train, y_train = x[:num_train], y[:num_train]
    # 校验集拥有从 num_train 起长度为 num_val 的样本
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # 测试集拥有尾部 num_test 个样本
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    # 返回这些集合
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = generate_training_data(x, y)
print('x_train.shape : ', x_train.shape)
print('y_train.shape : ', y_train.shape)
print('x_val.shape : ', x_val.shape)
print('y_val.shape : ', y_val.shape)
print('x_test.shape : ', x_test.shape)
print('y_test.shape : ', y_test.shape)

import torch
# 获取数据中的 x, y
x,y = generate_data(data)

# 将 x,y 转换乘 tensor ， Pytorch 模型默认的类型是 float32
x = torch.tensor(x)
y = torch.tensor(y)

print(x.shape,y.shape)

# 将 y 转化形状
y = y.view(y.shape[0],1)

print(x.shape,y.shape)
# 对 x, y 进行 minmaxscale
x_scaled = scaler.transform(x.reshape(-1,1)).reshape(-1,14)
y_scaled = scaler.transform(y)

x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
# 处理出训练集，校验集和测试集
x_train, y_train, x_val, y_val, x_test, y_test = generate_training_data(x_scaled, y_scaled)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def __getitem__(self, item):
        return self.x[item], self.y[item]
 
    def __len__(self):
        return len(self.x)
train_data = MyDataset(x_train,y_train)
valid_data = MyDataset(x_val,y_val)
test_data = MyDataset(x_test,y_test)

batch_size = 1

# 创建对应的 DataLoader
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 校验集和测试集的 shuffle 是没有必要的，因为每次都会全部跑一遍
valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
for i, read_data in enumerate(test_iter):
    # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    #print("第 {} 个Batch \n{}".format(i, read_data))
    break
# 表示输出数据
#print(read_data[0].shape, read_data[0])
# 表示输出标签
#print(read_data[1].shape, read_data[1])

# 使用均方根误差
loss = torch.nn.MSELoss()

# 自定义的损失函数，可以直接调用
def my_loss_func(y_hat, y):
    return compute_mae(y_hat, y)
model = LSTMNet(num_inputs,num_outputs)
# 使用 Adam 优化器， learning rate 调至 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_iter, valid_iter, loss, num_epochs, params=None, optimizer=None):
    
    # 用于绘图用的信息
    train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes = [], [], [], [], [], []
    
    # 循环 num_epochs 次
    for epoch in range(num_epochs):
        # 初始化参数
        train_l_sum, n = 0.0, 0
        # 初始化时间
        start = time.time()
        # 模型改为训练状态，如果使用了 dropout, batchnorm 之类的层时，训练状态和评估状态的表现会有巨大差别
        model.train()
        
        # 对训练数据集的每个 batch 执行
        for x, y in train_iter:
            
            # 如果使用了 GPU 则拷贝进显存
            if (use_gpu):
                x,y = x.cuda(),y.cuda()
            
            # 计算 y_hat
            y_hat = model(x)
            
            # 计算损失
            l = loss(y_hat, y).mean()

            # 梯度清零
            optimizer.zero_grad()
            
            # L1 正则化
            # for param in params:
            #     l += torch.sum(torch.abs(param))
            
            # L2 正则化可以在 optimizer 上加入 weight_decay 的方式加入

            # 求好对应的梯度
            l.backward()

            # 执行一次反向传播
            optimizer.step()

            # 对 loss 求和（在下面打印出来）
            train_l_sum += l.item() * y.shape[0]
            
            # 计数一共有多少个元素
            n += y.shape[0]
            
        # 模型开启预测状态
        model.eval()
        
        # 同样的，我们可以计算验证集上的 loss
        valid_l_sum, valid_n = 0, 0
        for x, y in valid_iter:
            # 如果使用了 GPU 则拷贝进显存
            if (use_gpu):
                x,y = x.cuda(),y.cuda()
            
            # 计算 y_hat
            y_hat = model(x)
            
            # 计算损失
            l = loss(y_hat, y).mean()

            # 对 loss 求和（在下面打印出来）
            valid_l_sum += l.item() * y.shape[0]
            
            # 计数一共有多少个元素
            valid_n += y.shape[0]
        
        # 对验证集合求指标
        # 这里训练集其实可以在循环内高效地直接算出，这里为了代码的可读性牺牲了效率
        train_mae, train_mape = evaluate_accuracy(train_iter, model)
        valid_mae, valid_mape = evaluate_accuracy(valid_iter, model)
        if (epoch+1) % 10 == 0:
            print('epoch %d, train loss %.6f, valid loss %.6f, train mae %.6f, mape %.6f, valid mae %.6f,mape %.6f, time %.2f sec'
              % (epoch + 1, train_l_sum / n, valid_l_sum / valid_n, train_mae, train_mape, valid_mae, valid_mape, time.time() - start))
        
        # 记录绘图有关的信息
        train_losses.append(train_l_sum / n)
        valid_losses.append(valid_l_sum / valid_n)
        train_maes.append(train_mae)
        train_mapes.append(train_mape)
        valid_maes.append(valid_mae)
        valid_mapes.append(valid_mape)
    model_path = 'results/mymodel.pt'
    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 返回一个训练好的模型和用于绘图的集合
    return model, (train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes)
  
  
# 训练模型
model, (train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes) = train_model(model, train_iter, test_iter, loss, 200, model.parameters(), optimizer)


# 输入的数量是前 14 个交易日的收盘价
num_inputs = 14
# 输出是下一个交易日的收盘价
num_outputs = 1

# ------------------------- 请加载您最满意的模型网络结构 -----------------------------
# 读取模型
model = LSTMNet(num_inputs,num_outputs)
    
# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/mymodel.pt'
# 模型保存的位置，如果模型路径不同，请修改！！！
model_path = 'results/mymodel.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(test_x):
    '''
    对于给定的 x 预测未来的 y 。
    :param test_x: 给定的数据集合 x ，对于其中的每一个元素需要预测对应的 y 。e.g.:np.array([[6.69,6.72,6.52,6.66,6.74,6.55,6.35,6.14,6.18,6.17,5.72,5.78,5.69,5.67]]
    :return: test_y 对于每一个 test_x 中的元素，给出一个对应的预测值。e.g.:np.array([[0.0063614]])
    '''
    # test 的数目
    n_test = test_x.shape[0]
    
    test_y = None
    # --------------------------- 此处下方加入读入模型和预测相关代码 -------------------------------
    # 此处为 Notebook 模型示范，你可以根据自己数据处理方式进行改动
    scaler = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
    test_x = scaler.transform(test_x.reshape(-1, 1)).reshape(-1, 14)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    
    test_y = model(test_x)
    
    # 如果使用 MinMaxScaler 进行数据处理，预测后应使用下一句将预测值放缩到原范围内
    test_y = scaler.inverse_transform(test_y.detach().cpu())
    test_y = test_y.detach().cpu().numpy()
    
    
    
    # --------------------------- 此处上方加入读入模型和预测相关代码 -------------------------------
    
    # 保证输出的是一个 numpy 数组
    assert(type(test_y) == np.ndarray)
    
    # 保证 test_y 的 shape 正确
    assert(test_y.shape == (n_test, 1))
    
    return test_y
# 测试用例
model_test_x = np.array([[6.69,6.72,6.52,6.66,6.74,6.55,6.35,6.14,6.18,6.17,5.72,5.78,5.69,5.67]])
print(predict(test_x = model_test_x))

# 输入的数量是前 14 个交易日的收盘价
num_inputs = 14
# 输出是下一个交易日的收盘价
num_outputs = 1
# 隐藏层的个数
num_hiddens = 128 

# 建立一个稍微复杂的 LSTM 模型
class LSTMNet(torch.nn.Module):
    def __init__(self, num_hiddens, num_outputs):
        super(LSTMNet, self).__init__()
        self.hidden_size = num_hiddens
        # RNN 层，这里的 batch_first 指定传入的是 (批大小，序列长度，序列每个位置的大小)
        # 如果不指定其为 True，传入顺序应当是 (序列长度，批大小，序列每个位置的大小)
        self.rnn = torch.nn.LSTM(input_size=num_inputs, hidden_size=num_hiddens,batch_first=True)
        # 线性层
        self.dense = torch.nn.Linear(self.hidden_size, 256)
        self.dense2 = torch.nn.Linear(256,num_outputs)
        # dropout 层，这里的参数指 dropout 的概率
        self.dropout = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.5)
        # ReLU 层
        self.relu = torch.nn.ReLU()
    
    # 前向传播函数，这是一个拼接的过程，使用大量变量是为了避免混淆，不做过多讲解
    def forward(self, x): 
        # LSTM 层会传出其参数，这里用 _ 将其舍弃
        x = x.view(batch_size,len(x),num_inputs)
        h, _ = self.rnn(x)
        # LSTM 层会传出 (batch_size, 1, num_hiddens) 个参数，故需要 reshape 后丢入全连接层
        h_r = h.reshape(-1,self.hidden_size)
        h_d = self.dropout(h_r)
        y = self.dense(h_d)
        drop_y = self.dropout2(y)
        a = self.relu(drop_y)
        y2 = self.dense2(a)
        return y2
# 判断 gpu 是否可用
use_gpu = torch.cuda.is_available()

def compute_mae(y_hat, y): 
    '''
    :param y: 标准值
    :param y_hat: 用户的预测值
    :return: MAE 平均绝对误差 mean(|y*-y|)
    '''
    return torch.mean(torch.abs(y_hat - y))

def compute_mape(y_hat, y):
    '''
    :param y: 标准值
    :param y_hat: 用户的预测值
    :return: MAPE 平均百分比误差 mean(|y*-y|/y)
    '''
    return torch.mean(torch.abs(y_hat - y)/y)

def evaluate_accuracy(data_iter, model): 
  	'''
    :param data_iter: 输入的 DataLoader
    :param model: 用户的模型
    :return: 对应的 MAE 和 MAPE
    '''
    # 初始化参数
    mae_sum, mape_sum, n = 0.0, 0.0, 0
    
    # 对每一个 data_iter 的每一个 x,y 进行计算
    for x, y in data_iter:
        
        # 如果运行在 GPU 上，需要将内存中的 x 拷贝到显存中
        if (use_gpu):
            x=x.cuda()
            y=y.cuda()
            
        # 计算模型得出的 y_hat
        y_hat = model(x)
        
        # 将 y_hat 逆归一化，这里逆归一化需要将数据转移到 CPU 才可以进行
        y_hat_real = torch.from_numpy(scaler.inverse_transform(np.array(y_hat.detach().cpu()).reshape(-1,1)).reshape(y_hat.shape))
        y_real = torch.from_numpy(scaler.inverse_transform(np.array(y.reshape(-1,1))).reshape(y.shape))
        
        # 计算对应的 MAE 和 RMSE 对应的和，并乘以 batch 大小
        mae_sum += compute_mae(y_hat_real,y_real) * y.shape[0]
        mape_sum += compute_mape(y_hat_real,y_real) * y.shape[0]
        
        # n 用于统计 DataLoader 中一共有多少数量
        n += y.shape[0]
        
    # 返回时需要除以 batch 大小，得到平均值
    return mae_sum / n, mape_sum / n
