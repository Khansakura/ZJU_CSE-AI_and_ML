#本代码省略部分语句，不可直接运行，仅用于参数辨识
import warnings
# 忽视警告
warnings.filterwarnings('ignore')
import os
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils,get_file
from keras_py.utils import get_random_data
from keras_py.face_rec import mask_rec
from keras_py.face_rec import face_rec
from keras_py.mobileNet import MobileNet
# 导入图片生成器
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 1.加载数据并进行数据处理
#加载数据集
basic_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/"
#图片尺寸的调整函数
def letterbox_image(image, size):
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image
#数据处理函数，参考keras的模块，详情参考https://keras-cn.readthedocs.io/en/latest/preprocessing/image/
def processing_data(data_path, height, width, batch_size=32,test_split=0.1):
    train_data=ImageDataGenerator(
        #浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        width_shift_range=0.1,
        #浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1,
        #浮点数，剪切强度（逆时针方向的剪切变换角度）
        shear_range=0.1,
        #浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        zoom_range=0.1,
        #浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        cval=0.1,
        #布尔值，进行随机水平翻转
        horizontal_flip=True,
        #布尔值，进行随机竖直翻转
        vertical_flip=True,
        #重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
        rescale=1./255,
        #数据集划分比例
        validation_split=test_split
    )
    #生成测试集
    test_data=ImageDataGenerator(
        rescale=1./255,
        validation_split=test_split
    )
    #训练集数据结构处理方式
    train_generator = train_data.flow_from_directory(
        #提供目录下需要有子目录
        data_path,
        #处理目标尺寸
        target_size=(height, width),
        #处理批次
        batch_size=batch_size,
        # "categorical", "binary", "sparse", "input" 或 None 之一。
        # 默认："categorical",返回one-hot 编码标签。
        class_mode='categorical',
        # 数据子集 ("training" 或 "validation")
        subset="training",
        seed=0
    )
    #测试集数据结构处理方式
    test_generator = test_data.flow_from_directory(
        data_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        subset="validation",
        seed=0
    )
    return train_generator, test_generator

#数据集路径
data_path = basic_path + 'image' 
#设置图片参数
height, width = 160, 160
#获取训练集和数据集
train_generator, test_generator = processing_data(data_path, height, width)
print('数据预处理完成...')
#获取文件夹属性名
labels = train_generator.class_indices
print(labels)
#字典键值对互换
labels = dict((v, k) for k, v in labels.items())
print(labels)


# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载
#加载文件
pnet_path = "/datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/pnet.h5"
rnet_path = "/datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/rnet.h5"
onet_path = "/datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/onet.h5"

# 加载 MobileNet 的预训练模型权
weight_path=basic_path+'keras_model_data/mobilenet_1_0_224_tf_no_top.h5'
height, width = 160, 160
model = MobileNet(input_shape=[height,width,3],classes=2)
model.load_weights(weight_path,by_name=True)
print('预训练模型加载完成...')

# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹
def save_model(model,save_path,model_dir):
    if os.path.exists(save_path):
        print("模型加载中")
        model.load_weights(save_path)
        print("模型加载完毕")
    checkpoint_period = ModelCheckpoint(
        # 模型存储路径
        model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        # 检测的指标
        monitor='val_accuracy',
        # ‘auto’，‘min’，‘max’中选择
        mode='max',
        # 是否只存储模型权重
        save_weights_only=False,
        # 是否只保存最优的模型
        save_best_only=True,
        # 检测的轮数是每隔1轮
        period=2
    )
    return checkpoint_period
checkpoint_save_path = "./results/temp1.h5"
model_dir = "./results/"
checkpoint_period = save_model(model, checkpoint_save_path, model_dir)
# 学习率下降的方式，acc三次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
                        monitor='accuracy',  # 检测的指标
                        factor=0.5,     # 当acc不下降时将学习率下调的比例
                        patience=3,     # 检测轮数是每隔三轮
                        verbose=2       # 信息展示模式
                    )
early_stopping = EarlyStopping(
                            monitor='val_accuracy',  # 检测的指标
                            min_delta=0.0001,         # 增大或减小的阈值
                            patience=10,         # 检测的轮数频率
                            verbose=1            # 信息展示的模式
                        )
# 一次的训练集大小
batch_size = 64
# 图片数据路径
data_path = basic_path + 'image'
# 图片处理
train_generator,test_generator = processing_data(data_path, height=160, width=160, batch_size=batch_size, test_split=0.1)
# 编译模型
model.compile(loss='binary_crossentropy',  # 二分类损失函数   
              optimizer=Adam(lr=5e-6),            # 优化器
              metrics=['accuracy'])        # 优化目标
# 训练模型
history = model.fit(train_generator,    
                    epochs=20, # epochs: 整数，数据的迭代总轮数。
                    # 一个epoch包含的步数,通常应该等于你的数据集的样本数量除以批量大小。
                    #steps_per_epoch=max(1,641 // batch_size),
                    steps_per_epoch = 641 // batch_size,
                    validation_data=test_generator,
                    #validation_steps=max(1,71 // batch_size),
                    validation_steps = 71 // batch_size,
                    initial_epoch=0, # 整数。开始训练的轮次（有助于恢复之前的训练）。
                    callbacks=[checkpoint_period, reduce_lr])
# 保存模型
model.save_weights(model_dir + 'temp.h5')
plt.plot(history.history['loss'],label = 'train_loss')
plt.plot(history.history['val_loss'],'r',label = 'val_loss')


# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],label = 'acc')
plt.plot(history.history['val_accuracy'],'r',label = 'val_acc')
plt.legend()
plt.show()

from keras_py.utils import get_random_data
from keras_py.face_rec import mask_rec
from keras_py.face_rec import face_rec
from keras_py.mobileNet import MobileNet
from PIL import Image
import cv2

# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/temp.h5'
model_path = 'results/temp.h5'
# ---------------------------------------------------------------------------

def predict(img):
    """
    加载模型和模型预测
    :param img: cv2.imread 图像
    :return: 预测的图片中的总人数、其中佩戴口罩的人数
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 将 cv2.imread 图像转化为 PIL.Image 图像，用来兼容测试输入的 cv2 读取的图像（勿删！！！）
    # cv2.imread 读取图像的类型是 numpy.ndarray
    # PIL.Image.open 读取图像的类型是 PIL.JpegImagePlugin.JpegImageFile
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    detect = mask_rec(model_path)
    img, all_num, mask_num = detect.recognize(img)
    
    # -------------------------------------------------------------------------
    return all_num,mask_num
