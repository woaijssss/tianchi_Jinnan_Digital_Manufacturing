import json
import cv2

import src.utils as utils

dir_name = '../datas/category_id_1/'


def imageCorrection(image_map, category_id_name):
    img_file = dir_name + image_map['file_name']
    color = (0, 255, 0)
    x = int(image_map['x'])
    y = int(image_map['y'])
    w = int(image_map['w'])
    h = int(image_map['h'])

    img = cv2.imread(img_file)
    # frame_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame_gray = img
    img_tmp = frame_gray[y:y + h, x:x + w]

    cv2.imshow('image_', frame_gray)
    cv2.imshow('image1', img_tmp)

    while True:
        c = cv2.waitKey(0)

        if c & 0xFF == ord('q'):
            quit()


if __name__ == '__main__':
    # 仅修改这个值，对应到特征id即可
    category_id = 1
    category_id_name = "category_id_" + str(category_id)

    filename = '../datas/' + category_id_name + '.json'

    with open(filename, 'r', encoding='utf-8') as f:
        json_str = json.load(f)

        for i in range(0, len(json_str['images'])):
            image_1_map = json_str['images'][i]
            if image_1_map['file_name'] == "190125_180619_00174593.jpg":
                imageCorrection(image_1_map, category_id_name)
                break