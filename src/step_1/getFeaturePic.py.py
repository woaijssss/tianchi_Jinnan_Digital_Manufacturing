
import json
import cv2

dir_name = "../jinnan2_round1_train_20190222/"

def imageTest(json_str):
    img_file = "../jinnan2_round1_train_20190222/restricted/190118_202347_00164778.jpg"
    img = cv2.imread(img_file)
    
    print(img.shape)

    # cv2.rectangle(img, (388, 207), (388+53, 207+56), (0, 255, 0), thickness=2)
    cv2.rectangle(img, (298, 266), (298+106, 266+125), (0, 255, 0), thickness=2)
    cv2.imshow("image Test", img)
    
    cv2.waitKey(0)
    
    print(json_str.keys())

if __name__ == '__main__':
    filename = "../train_no_poly.json"
    
    with open(filename, 'r') as fd:
        line = fd.readlines()[0]
    
    json_str = json.loads(line)
    
    imageTest(json_str)