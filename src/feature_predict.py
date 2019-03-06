
import src.utils as utils
import src.contours as contours
from lib.ModelFit import CNNModel
import src.submit as submit
import numpy as np

import cv2

def imgShow(img, c):
    x, y, w, h = cv2.boundingRect(c)  # 将轮廓分解为识别对象的左上角坐标和宽、高
    # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
    cv2.imshow("detectTarget", img)
    
    while True:
        key = chr(cv2.waitKey(15) & 255)
        if key == 'q':
            cv2.destroyAllWindows()
            break

'''
    找到图中预测各类别概率的最大值，并返回为list
'''
def findMaxProb(contours_set):
    # p1, p2, p3, p4, p5 = [], [], [], [], []
    prob_lst = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
    prob = []
    for c in contours_set:
        x, y, w, h = cv2.boundingRect(c)
        img_tmp = blur[y:y+h+10, x:x+w+10]
        id, result = model.predict(img_tmp)
        prob.append(result)

    if len(prob):
        arr = np.array(prob)
        prob_lst[0] = "%.2f" % arr[:, 0].max()
        prob_lst[1] = "%.2f" % arr[:, 1].max()
        prob_lst[2] = "%.2f" % arr[:, 2].max()
        prob_lst[3] = "%.2f" % arr[:, 3].max()
        prob_lst[4] = "%.2f" % arr[:, 4].max()

    return prob_lst

def test():
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print(arr.max())
    quit()

if __name__ == '__main__':
    # test()
    model_path = "./step_2/cnnmodel.h5"
    
    # 加载模型
    model = CNNModel()
    model.loadModel(path=model_path)

    dir_name = "../../jinnan2_round1_test_a_20190222/"
    submit = submit.Submit()
    image_lst = submit.getAllPictures(dir_name)[0]      # 获取测试集目录下所有的图片文件名

    i = 0
    for image in image_lst:     # 读取所有图片
        i += 1
        pic = dir_name + image
        img = cv2.imread(pic)

        # 高斯滤波,降低噪声
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        contours_set = contours.targetDetect(img, (100, 255))

        prob_lst = findMaxProb(contours_set)        # 找到图中预测各类别概率的最大值，并返回为list
        # prob_lst = list(np.array(prob_lst, dtype=np.float))
        # print(prob_lst)
        submit.getSubmitJson(image, prob_lst)
        # imgShow(img, c)

        print('第 %d 张图片' % i)

    print(submit.get())
    submit.save()
        