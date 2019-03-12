
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
            
def getMap(c, label=1, confidence=0.0):
    ret = {}
    x, y, w, h = cv2.boundingRect(c)
    ret["xmin"] = x
    ret["xmax"] = x+w
    ret["ymin"] = y
    ret["ymax"] = y+h
    ret["label"] = label
    ret["confidence"] = float('%.2f' % confidence)
    
    return ret

'''
    找到图中预测各类别概率的最大值，并返回为list
'''
def findMaxProb(contours_set):
    # p1, p2, p3, p4, p5 = [], [], [], [], []
    # prob_lst = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
    prob_lst = []
    prob = []
    contour_list = []
    print('contours_set len: ', len(contours_set))
    for c in contours_set:          # 遍历每个轮廓
        x, y, w, h = cv2.boundingRect(c)
        img_tmp = blur[y:y+h+10, x:x+w+10]
        id, result = model.predict(img_tmp)     # 预测结果: id 和与其对应的结果集
        prob.append(result)
        contour_list.append((id, c))

    if len(prob):
        arr = np.array(prob)
        '''
            每一个类别最大的概率
        '''
        feature_lst_1 = list(arr[:, 0])
        feature_lst_2 = list(arr[:, 1])
        feature_lst_3 = list(arr[:, 2])
        feature_lst_4 = list(arr[:, 3])
        feature_lst_5 = list(arr[:, 4])
        confidence_1 = max(feature_lst_1)
        confidence_2 = max(feature_lst_2)
        confidence_3 = max(feature_lst_3)
        confidence_4 = max(feature_lst_4)
        confidence_5 = max(feature_lst_5)
        index_1 = feature_lst_1.index(confidence_1)
        index_2 = feature_lst_2.index(confidence_2)
        index_3 = feature_lst_3.index(confidence_3)
        index_4 = feature_lst_4.index(confidence_4)
        index_5 = feature_lst_5.index(confidence_5)
        
        pic_1 = getMap(contour_list[index_1][1], 1, confidence_1)
        pic_2 = getMap(contour_list[index_2][1], 2, confidence_2)
        pic_3 = getMap(contour_list[index_3][1], 3, confidence_3)
        pic_4 = getMap(contour_list[index_4][1], 4, confidence_4)
        pic_5 = getMap(contour_list[index_5][1], 5, confidence_5)

        print('----pic_1: ', pic_1)
        print('----pic_2: ', pic_2)
        print('----pic_3: ', pic_3)
        print('----pic_4: ', pic_4)
        print('----pic_5: ', pic_5)
        
        # prob_lst["rects"] = [pic_1, pic_2, pic_3, pic_4, pic_5]
        prob_lst = [pic_1, pic_2, pic_3, pic_4, pic_5]

    return prob_lst

def test():
    import numpy as np
    lst = [
        [1.1123123, 2, 3, 4, 5],
        [10.1333341, 2, 3, 4, 5],
        [11.112323334444, 2, 3, 4, 5],
        [12.1123234455666, 2, 3, 4, 5]
    ]
    
    arr = np.array(lst, dtype=np.float32)
    print(arr)
    # li1, li2, li3, li4, li5 = arr[:, 0:5:1]
    li1 = list(arr[:, 0])
    li2 = list(arr[:, 1])
    li3 = list(arr[:, 2])
    li4 = list(arr[:, 3])
    li5 = arr[:, 4]
    print(li1)
    print(li2)
    print(li3)
    print(li4)
    print(li5)
    max_v = max(li1)
    print(max_v)
    print(li1.index(max_v))
    # print(list(arr[:, 0]).index(arr[:, 0].max().index))
    
    quit()

if __name__ == '__main__':
    # test()
    model_path = "../src/step_2/cnnmodel.h5"
    
    # 加载模型
    model = CNNModel()
    model.loadModel(path=model_path)

    dir_name = "../../jinnan2_round1_test_a_20190306/"
    submit = submit.Submit()
    # image_lst = submit.getAllPictures(dir_name)[0]      # 获取测试集目录下所有的图片文件名

    image_lst = []
    with open("empty_imgs.txt", "r+") as fd:
        lines = fd.readlines()
        for line in lines:
            image_lst.append(line.strip("\n"))

    i = 0
    for image in image_lst:     # 读取所有图片
        i += 1
        # if i == 2:
        #     break
        pic = dir_name + image
        img = cv2.imread(pic)

        # 高斯滤波,降低噪声
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        contours_set = contours.targetDetect(img, (5, 230))

        sub = findMaxProb(contours_set)        # 找到图中预测各类别概率的最大值，并返回为list
        # print(sub)
        # prob_lst = list(np.array(prob_lst, dtype=np.float))
        # print(prob_lst)
        obj = submit.getSubmitJson(image, sub)
        # imgShow(img, c)

        print('第 %d 张图片' % i, "------", obj)
    #
    print(submit.get())
    submit.save()









    
    
    
    
    
    
    
    
    
    
    
    
        