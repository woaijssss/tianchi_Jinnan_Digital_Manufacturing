#coding:utf-8

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.applications import resnet50

from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数

from lib.LoadDataSet import resizeImg, IMAGE_SIZE
from tensorflow.python.client import device_lib

# CNN网络模型类
class CNNModel:
	def __init__(self):
		self.model = None

	# 建立模型
	def buildModel(self, data_set, nb_classes = 5):
		print('===>:', data_set.input_shape)
		# 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加
		# 专称：序贯模型或线性堆叠模型:多个网络层的线性堆叠，也就是“一条路走到黑”。
		self.model = Sequential()

		weight_decay = 0.0005
		nb_epoch = 100
		batch_size = 32

		##########################################################################
		# 第一个 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，stride没写，默认应该是1*1
		# 对于stride=1*1,并且padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
		self.model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=data_set.input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		# 进行一次归一化
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.3))
		# layer2 32*32*64
		self.model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		# 下面两行代码是等价的，#keras Pool层有个奇怪的地方，stride,默认是(2*2),
		# padding默认是valid，在写代码是这些参数还是最好都加上,这一步之后,输出的shape是16*16*64
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
		# layer3 16*16*64
		self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer4 16*16*128
		self.model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		# layer5 8*8*128
		self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer6 8*8*256
		self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer7 8*8*256
		self.model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		# layer8 4*4*256
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer9 4*4*512
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer10 4*4*512
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		# layer11 2*2*512
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer12 2*2*512
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.4))
		# layer13 2*2*512
		self.model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.5))
		# layer14 1*1*512
		'''
		flatten:是把池化层展开以便作为全连接层的输入。
		"展开"即：指的是将一个(m, n)的池化后的矩阵，转化为(m*n, 1)的矩阵。
		'''
		self.model.add(Flatten())
		self.model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		# layer15 512
		self.model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		# layer16 512
		self.model.add(Dropout(0.5))
		self.model.add(Dense(nb_classes))
		self.model.add(Activation('softmax'))
		##########################################################################

		# 输出模型的情况
		'''
		这个网络模型共18层，包括:
		4个卷积层
		5个激活函数层
		2个池化层（pooling layer）
		3个Dropout层
		2个全连接层
		1个Flatten层
		1个分类层
		训练参数为6,489,634个。
		'''
		self.model.summary()

	# 训练模型
	'''
	:param nb_epoch:训练轮数
	'''
	def trainModel(self, data_set, batch_size=20, nb_epoch=1, data_arg=True):
		# 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy',	# 损失函数
						   optimizer=sgd,
						   metrics=['accuracy'])	# 完成实际的模型配置工作

		'''
		不使用数据提升；
		所谓的提升就是从训练数据中利用旋转、翻转、加噪声等方法创造新的训练集，
		有意识的提升训练数据模型，增加模型训练量。
		'''
		if not data_arg:
			self.model.fit(
				data_set.train_images,
				data_set.train_labels,
				batch_size=batch_size,
				nb_epoch=nb_epoch,
				validation_data=(data_set.valid_images, data_set.valid_labels),
				shuffle=True
			)
		# 使用实时数据提升
		else:
			'''
			定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用
			一次，其生成一组数据(顺序生成)，节省内存，其实就是python的生成器和迭代器。
			'''
			datagen = ImageDataGenerator(
				featurewise_center=False,  				# 是否使输入数据去中心化（均值为0），
				samplewise_center=False,  				# 是否使输入数据的每个样本均值为0
				featurewise_std_normalization=False,  	# 是否数据标准化（输入数据除以数据集的标准差）
				samplewise_std_normalization=False,  	# 是否将每个样本数据除以自身的标准差
				zca_whitening=False,  					# 是否对输入数据施以ZCA白化
				rotation_range=20,  					# 数据提升时图片随机转动的角度(范围为0～180)
				width_shift_range=0.2,  				# 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
				height_shift_range=0.2,  				# 同上，只不过这里是垂直
				horizontal_flip=True,  					# 是否进行随机水平翻转
				vertical_flip=False  					# 是否进行随机垂直翻转
			)

			# 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
			datagen.fit(data_set.train_images)

			print(data_set.valid_images.shape)
			print(data_set.valid_labels.shape)
			print(data_set.valid_labels)

			# 利用生成器开始训练模型
			self.model.fit_generator(
				datagen.flow(
					data_set.train_images, data_set.train_labels, batch_size=batch_size
				),
				samples_per_epoch=data_set.train_images.shape[0],
				nb_epoch=nb_epoch,
				validation_data=(data_set.valid_images, data_set.valid_labels)
			)

	# 不加.h5会提示:ImportError: `save_model` requires h5py.错误
	def saveModel(self, path='./cnnmodel.h5'):
		self.model.save(path)

	def loadModel(self, path='./cnnmodel.h5'):
		self.model = load_model(path)

	def evaluate(self, data_set):
		score = self.model.evaluate(
			data_set.test_images, data_set.test_labels, verbose=1
		)
		print('%s: %.2f%%' % (self.model.metrics_names[1], score[1]*100))

	# 识别人脸，判断是不是我自己本人
	'''
	这个函数是提供给外部模块使用的，外部模块用它来预测哪个是“我”，哪个不是“我”。
	'''
	def predict(self, image):
		# 依然是根据后端系统确定维度顺序
		if K.image_dim_ordering() == 'th'\
			and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
			image = resizeImg(image)	# 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
			image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))	# 与模型训练不同，这次只是针对1张图片进行预测
		elif K.image_dim_ordering() == 'tf'\
			and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
			image = resizeImg(image)
			image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

		# 浮点并归一化
		image = image.astype('float32')
		image /= 255

		# 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各位多少
		result = self.model.predict_proba(image)
		# print('result: ', result)

		# 给出类别预测：0或1
		id = self.model.predict_classes(image)

		# 返回类别预测结果
		return id, result[0]

if __name__ == '__main__':
	from src.DataSetHandling import DataSet
	data_set = DataSet('./data')
	data_set.load()

	model = CNNModel()
	#model.buildModel(data_set)

	# 测试训练函数的代码
	'''
	训练误差：loss: 1.1921e-07
	训练准确率：acc: 1.0000
	验证误差：val_loss: 1.1921e-07
	验证准确率：val_acc: 1.0000
	'''
	#model.trainModel(data_set)
	#model.saveModel()

	# 用测试集评估模型
	model.loadModel()
	model.evaluate(data_set)
















