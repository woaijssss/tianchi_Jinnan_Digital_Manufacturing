
from lib.DataSetHandling import DataSet
from lib.ModelFit import CNNModel

import src.utils as utils

'''
    将category_id_1_featrue_gray和category_id_2_featrue_gray这两个目录，拷贝到test_picture目录下
    该程序自动创建test_picture
'''

if __name__ == '__main__':
    directory = '../../datas/test_pictures'

    utils.mkdir(directory)

    data_set = DataSet(directory)
    data_set.load()
    
    model = CNNModel()
    model.buildModel(data_set)
    
    # 测试训练函数的代码
    model.trainModel(data_set)
    model.saveModel()

    model.evaluate(data_set)