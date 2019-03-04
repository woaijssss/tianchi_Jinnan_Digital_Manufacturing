# -*- coding:utf-8 -*-

import numpy as np
import argparse
import cv2

def test():
    filename = "E:\比赛\tianchi\tianchi_Jinnan_Digital_Manufacturing\datas\test_pictures\category_id_1_feature_gray\190109_182654_00154240.jpg".replace('\\', "\\\\")
    print(filename)
    img = cv2.imread(filename)
    print(img)
    quit()


def edge(img):
    # 高斯模糊,降低噪声
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # 灰度图像
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # 图像梯度
    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # 计算边缘
    # 50和150参数必须符合1：3或者1：2
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)

    return edge_output
    # 图一
    # cv2.imshow("edge", edge_output)
    #
    # dst = cv2.bitwise_and(img, img, mask=edge_output)
    # # 图二（彩色）
    # cv2.imshow('cedge', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

class CompareImage():

    def compare_image(self, path_image1, path_image2):

        # imageA = cv2.imread(path_image1)
        # imageB = cv2.imread(path_image2)

        # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        from skimage.measure import compare_ssim

        (score, diff) = compare_ssim(path_image1, path_image2, full=True)
        print("SSIM: {}".format(score))
        return score

if __name__ == '__main__':
    print(cv2.getVersionString())
    quit()
    pic_name = "../datas/category_id_1/190119_175805_00166370.jpg"
    img = cv2.imread(pic_name)  # 读取图片

    kernel_2 = np.ones((2, 2), np.uint8)    # 2*2卷积核
    kernel_3 = np.ones((3, 3), np.uint8)    # 3*3卷积核
    kernel_4 = np.ones((4, 4), np.uint8)    # 4*4卷积核

    if img is not None:     # 判断图片是否读入
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      # 把BGR图像转换为HSV格式
        '''
            HSV模型中颜色的参数分别是：色调（H），饱和度（S），名都（V）
            下面两个值是要识别的颜色范围
        '''
        # lower = np.array([20, 20, 20])      # 要识别颜色的下限
        # upper = np.array([30, 255, 255])    # 要识别颜色的上限

        '''
            HSV图像过滤通道（反色）
        '''
        # lower = np.array([0, 0, 240])  # 要识别颜色的下限
        # upper = np.array([100, 120, 255])    # 要识别颜色的上限

        '''
            RGB图像过滤通道
        '''
        lower = np.array([40, 30, 0])  # 要识别颜色的下限
        upper = np.array([210, 230, 210])    # 要识别颜色的上限


        # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
        mask = cv2.inRange(img, lower, upper)

        '''
            下面四行是用卷积进行滤波：
            erosion： 侵蚀
            dilation： 扩张
        '''
        erosion = cv2.erode(mask, kernel_3, iterations=1)
        erosion = cv2.erode(erosion, kernel_3, iterations=1)
        dilation = cv2.dilate(erosion, kernel_3, iterations=1)
        dilation = cv2.dilate(dilation, kernel_3, iterations=1)

        # target是把原图中的非目标颜色区域去掉剩下的图像
        target = cv2.bitwise_and(img, img, mask=dilation)

        # dilation = edge(img)

        # for i in range(1, 255):
        # 将滤波后的图像变成二值图像放在binary中
        ret, binary = cv2.threshold(dilation, 0, 10, cv2.THRESH_BINARY)

        # 在binary中发现轮廓，卢阔按照面积从小到大排列
        # binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        p = 0
        # print(len(contours))
        for i in contours:  # 遍历所有的轮廓
            x, y, w, h = cv2.boundingRect(i)    # 将轮廓分解为识别对象的左上角坐标和宽、高
            # 在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, ), 2)
            # 给识别对象协商标号
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(p), (x-10, y+10), font, 1, (0, 0, 255), 2)     # 加减10是调整字符位置
            p += 1

            img_tmp = img[y:y + h, x:x + w]
            ci = CompareImage()
            print(img.shape)
            print(img_tmp.shape)
            # ci.compare_image(img_tmp, img)

        cv2.imshow('target', target)
        cv2.imshow('Mask', mask)
        cv2.imshow('prod', dilation)
        cv2.imshow('img', img)
        cv2.imwrite('img.jpg', img) # 将画上矩形的图形保存到当前目录

        while True:
            key = chr(cv2.waitKey(15) & 255)
            if key == 'q':
                cv2.destroyAllWindows()
                break


