
import src.utils as utils
from lib.ModelFit import CNNModel

if __name__ == '__main__':
    model_path = "./step_2/cnnmodel.h5"
    
    # 加载模型
    model = CNNModel()
    model.loadModel(path=model_path)
    
    
    dir_name = "../../jinnan2_round1_test_a_20190222/"