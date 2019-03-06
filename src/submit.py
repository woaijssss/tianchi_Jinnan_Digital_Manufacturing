
import os
import json
import src.utils as utils
import src.contours as contours

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

class Submit:
    submit_json = {}
    def __init__(self):
        pass

    def getAllPictures(self, dir_name):
        image_lst = []

        for root, dirs, files in os.walk(dir_name):
            image_lst.append(files)

        return image_lst

    def getSubmitJson(self, img_name, prob_lst=[0.0, 0.0, 0.0, 0.0, 0.0]):
        dict_tmp = {}
        dict_tmp["1"] = prob_lst[0]
        dict_tmp["2"] = prob_lst[1]
        dict_tmp["3"] = prob_lst[2]
        dict_tmp["4"] = prob_lst[3]
        dict_tmp["5"] = prob_lst[4]
        self.submit_json[img_name] = dict_tmp

    def get(self):
        return json.dumps(self.submit_json)

    def save(self, filename='./jinnan2_round1_submit_20190222.json'):
        with open(filename, 'w+') as fd:
            fd.write(json.dumps(self.submit_json))

if __name__ == '__main__':
    dir_name = "../../jinnan2_round1_test_a_20190222/"
    submit = Submit()
    image_lst = submit.getAllPictures(dir_name)[0]

    for image in image_lst:
        pic = dir_name + image
        img = cv2.imread(pic)
        # 高斯滤波,降低噪声
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        contours_set = contours.targetDetect(img, (100, 255))

        # imgShow(img, contours_set[0])

        if contours_set is None:
            continue

        for c in contours_set:
            x, y, w, h = cv2.boundingRect(c)
            img_tmp = blur[y:y + h + 10, x:x + w + 10]

            # id, result = model.facePredict(img_tmp)

            # imgShow(img, c)

            # submit_json = json.dumps(submit_json)
            submit.getSubmitJson(image)

    print(submit.get())
    print(type(submit.get()))
    submit.save()

