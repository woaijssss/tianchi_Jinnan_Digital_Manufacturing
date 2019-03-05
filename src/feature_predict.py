
import src.utils as utils
import src.contours as contours
from lib.ModelFit import CNNModel

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

if __name__ == '__main__':
    model_path = "./step_2/cnnmodel.h5"
    
    # 加载模型
    model = CNNModel()
    model.loadModel(path=model_path)
    
    
    # dir_name = "../../jinnan2_round1_test_a_20190222/"
    dir_name = "../datas/category_id_1/"
    pic = dir_name + "190122_182038_00169899.jpg"
    img = cv2.imread(pic)
    # 高斯滤波,降低噪声
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    contours_set = contours.targetDetect(img, (150, 255))
    
    # imgShow(img, contours_set[0])

    for c in contours_set:
        x, y, w, h = cv2.boundingRect(c)
        img_tmp = blur[y:y+h+10, x:x+w+10]
        
        id = model.facePredict(img_tmp)
        
        if not id:
            imgShow(img, c)
            break
            
        