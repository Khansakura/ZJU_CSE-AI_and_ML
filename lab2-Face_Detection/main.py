#本代码不可直接运行，省略了部分语句，仅体现重要过程及参数选择
# 导入相关包
import glob, os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import random,cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import time
# 导入图片生成器
from tensorflow.keras.preprocessing.image import ImageDataGenerator

						# 对图片的每个像素值均乘上这个放缩因子，把像素值放缩到0和1之间有利于模型的						 收敛
            rescale=1. / 255,  
            # 浮点数，剪切强度（逆时针方向的剪切变换角度）
            shear_range=0.1,  
            # 随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 -zoom_range, 														1+zoom_range]
            zoom_range=0.1,
            # 浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
            width_shift_range=0.1,
            # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
            height_shift_range=0.1,
            # 布尔值，进行随机水平翻转
            horizontal_flip=True,
            # 布尔值，进行随机竖直翻转
            vertical_flip=True,
            # 在 0 和 1 之间浮动。用作验证集的训练数据的比例
            validation_split=validation_split  
            
 				#创建一个深度学习模型，
				model = Sequential()
				#第一个卷积块
        model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', input_shape=(25, 25, 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same'))
        model.add(Dropout(0.2))
 
        # #第二个卷积块
        model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))
        model.add(Dropout(0.5))
 
        #第三个卷积块
        model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same'))
        model.add(Dropout(0.4))
 
 
        #将上一层的输出特征映射转化为一维数据，以便进行全连接操作
        model.add(Flatten())
 
        # #第一个全连接层
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
 
        #第二个全连接层
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.2))
 
        #第三个全连接层
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.4))
        
        #第四个全连接层
        model.add(Dense(10,activation='softmax'))
				# 打印模型概况
				model.summary()
       
      
				sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
        opt = Adam(learning_rate=1e-4)
        
        rmsprop = RMSprop(lr=0.001,rho=1,epsilon=None,decay=0)
  
  			# 编译模型, 采用 compile 函数: https://keras.io/models/model/#compile
        model.compile(
                # 是优化器, 主要有Adam、sgd、rmsprop等方式。
                optimizer=opt,
                # 损失函数,多分类采用 categorical_crossentropy
                loss='categorical_crossentropy',
                # 是除了损失函数值之外的特定指标, 分类问题一般都是准确率
                metrics=['accuracy'])

        # 可视化，TensorBoard 是由 Tensorflow 提供的一个可视化工具。
        tensorboard = TensorBoard(log_dir)
    
        # 训练模型, fit函数:https://keras.io/models/model/#fit
        # 利用Python的生成器，逐个生成数据的batch并进行训练。
        # callbacks: 实例列表。在训练时调用的一系列回调。详见 							   														https://keras.io/callbacks/。
        batch_size = 64
        d = model.fit(
                # 一个生成器或 Sequence 对象的实例
                x=train_generator,
                # epochs: 整数，数据的迭代总轮数。
                epochs=20,
                # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                steps_per_epoch=1620 // batch_size,
                # 验证集
                validation_data=validation_generator,
                # 在验证集上,一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                validation_steps=180 // batch_size,
                callbacks=[tensorboard])
        # 模型保存
        model.save(model_save_path)
  def evaluate_mode(test_data, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_data: 测试集数据
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    # 加载模型
    model = load_model('results/star.h5')
    # 测试集数据与标签
    test_x, test_y = test_data.__getitem__(2)
    # 预测值
    y = model.predict(test_x)
    # 绘制预测图像的预测值和真实值，定义画布
    plt.figure(figsize=(16, 16))
    labels = {0: 'CL', 1: 'FBB', 2: 'HG', 3: 'HJ', 4: 'LHR', 5: 'LSS', 6: 'LYF', 7: 'PYY', 8: 'TY', 9: 'YM'}
    for i in range(16):
        # 绘制各个子图
        plt.subplot(4, 4, i + 1)
        
        # 图片名称
        plt.title('pred:%s / truth:%s' % (labels[np.argmax(y[i])], labels[np.argmax(test_y[i])]))

        # 展示图片
        plt.imshow(test_x[i])
    
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

    mae_sum += compute_mae(y,test_y) * y.shape[0]
    mape_sum += compute_mape(y,test_y) * y.shape[0]
        
    # n 用于统计 DataLoader 中一共有多少数量
    n += y.shape[0]

    print(mae_sum / n, mape_sum / n)


    # 绘制模型训练过程的损失和平均损失
    # 绘制模型训练过程的损失值曲线，标签是 loss
    plt.plot(res.history['loss'], label='loss')
    print('{"metric": "loss", "value": float}')
    # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
    plt.plot(res.history['val_loss'], label='val_loss')
    print('{"metric": "val_loss", "value": float}')
    # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
    plt.legend(loc='upper right')
    # 展示图片
    plt.show()
    # 绘制模型训练过程中的的准确率和平均准确率
    # 绘制模型训练过程中的准确率曲线，标签是 acc
    plt.plot(res.history['accuracy'], label='accuracy')
    print('{"metric": "accuracy", "value": float}')
    # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
    plt.plot(res.history['val_accuracy'], label='val_accuracy')
    print('{"metric": "val_accuracy", "value": float}')
    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()
    # 展示图片
    plt.show()
    # ---------------------------------------------------------------------------
