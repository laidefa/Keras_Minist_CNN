# encoding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

###导入keras相关卷积模块，包含Dropout、Conv2D和MaxPoling2D
import numpy as np
from keras.datasets import mnist
import keras
import gc
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
time1 = time.time()


######读入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

##看一下数据集大小
# print(X_train[0].shape)
# print(y_train[0])

##把训练集中的手写黑白字体变成标准的四维张量形式(样本数量，长，宽，1)，并把像素值变成浮点格式。
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32') 
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')


####归一化：由于每个像素值都是介于0-255，所以这里统一除以255，把像素值控制在0~1范围。
X_train /= 255 
X_test /= 255


##由于输入层需要10个节点，所以最好把目标数字0-9做成one Hot编码的形式。
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe


########把标签用one Hot编码重新表示一下
y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_train_ohe = y_train_ohe.astype('float32')
y_test_ohe = y_test_ohe.astype('float32')


###接着搭建卷积神经网络

model = Sequential() 

###添加1层卷积层，构造64个过滤器，每个过滤器覆盖范围是3*3*1，过滤器挪动步长为1，图像四周补一圈0，并用relu 进行非线性变换
model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
          input_shape = (28,28,1)))

###添加1层Max pooling,在2*2的格子中取最大值

model.add(MaxPooling2D(pool_size = (2, 2)))

##设立Dropout层，将dropout的概率设为0.5。也可以尝试用0.2,0.3这些常用的值
model.add(Dropout(0.5))

##重复构造，搭建神经网络
model.add(Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.5)) 


model.add(Conv2D(256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.5))


###把当前层节点展平
model.add(Flatten())

######构造全连接神经网络层(3层)
model.add(Dense(128, activation = 'relu')) 
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(32, activation = 'relu')) 


model.add(Dense(10, activation = 'softmax'))


#定义损失函数，一般来说分类问题的损失函数都选择采用交叉熵(Crossentropy)
# 我们可以定制各种选项，比如下面就定制了优化器选项。
adamoptimizer = keras.optimizers.Adam(lr = 1e-4)
model.compile(loss = 'categorical_crossentropy', 
              optimizer = adamoptimizer, metrics = ['accuracy'])

######放入批量样本，进行训练
model.fit(X_train, y_train_ohe, validation_data = (X_test, y_test_ohe), 
          epochs = 20, batch_size = 128)


#######在测试集上评价模型精确度
scores=model.evaluate(X_test,y_test_ohe,verbose=0)

#####打印精确度
print scores


time2 = time.time()
print u'ok,结束!'
print u'总共耗时：' + str(time2 - time1) + 's'









