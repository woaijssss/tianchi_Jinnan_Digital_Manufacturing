# -*- coding:utf-8 -*-

import numpy as np
import argparse
import cv2

'''
    Canny算子计算边缘轮廓
'''
def edge(img):
    # 高斯滤波,降低噪声
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # 灰度转换
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    # 沿x、y方向分别计算图像梯度
    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    '''
        Canny算子计算边缘
        50和150参数必须符合1：3或者1：2
    '''
    edge_out = cv2.Canny(x_grad, y_grad, 50, 150)
    
    return edge_out

'''
    检测目标区域
    @param img: 图像帧
    @param threshold: 二值化的像素上下限
'''
def targetDetect(img, threshold=(0, 255)):
    if img is not None:  # 判断图片是否读入
        low, high = threshold
        
        kernel = np.ones((3, 3), np.uint8)  # 3*3卷积核
        
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把BGR图像转换为HSV格式
        
        '''
            RGB图像过滤通道
        '''
        lower = np.array([40, 30, 0])  # 要识别颜色的下限
        upper = np.array([210, 230, 210])  # 要识别颜色的上限
        
        # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
        mask = cv2.inRange(img, lower, upper)
        
        '''
            下面四行是用卷积进行滤波，形态学算法：
            erosion： 腐蚀
            dilation： 膨胀
        '''
        erosion = cv2.erode(mask, kernel, iterations=1)
        erosion = cv2.erode(erosion, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        dilation = cv2.dilate(dilation, kernel, iterations=1)
        
        # target是把原图中的非目标颜色区域去掉剩下的图像
        target = cv2.bitwise_and(img, img, mask=dilation)
        
        dilation = edge(target)
        
        ret, binary = cv2.threshold(dilation, low, high, cv2.THRESH_BINARY)
        
        # 在binary中发现轮廓，卢阔按照面积从小到大排列
        binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按照轮廓从大到小排序
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
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
    pic_name = "../../jinnan2_round1_test_a_20190222/190109_183554_00154272.jpg"
    img = cv2.imread(pic_name)  # 读取图片
    
    targetDetect(img, (0, 255))




