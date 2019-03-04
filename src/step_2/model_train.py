
from lib.DataSetHandling import DataSet
from lib.ModelFit import CNNModel

if __name__ == '__main__':
    data_set = DataSet('../../datas/test_pictures')
    data_set.load()
    
    model = CNNModel()
    model.buildModel(data_set)
    
    # 测试训练函数的代码
    '''
    训练误差：loss: 1.1921e-07
    训练准确率：acc: 1.0000
    验证误差：val_loss: 1.1921e-07
    验证准确率：val_acc: 1.0000
    '''
    model.trainModel(data_set)
    model.saveModel()