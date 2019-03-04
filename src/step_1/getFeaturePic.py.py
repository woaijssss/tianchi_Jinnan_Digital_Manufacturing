
import json
import cv2

'''
    将训练集目录下restricted所有category_id为0的图像取出，放入./datas/category_id_1目录中；
    并提取json标注中的对应部分，保存在 datas/category_id_1.json 中
'''

dir_name = "../../../jinnan2_round1_train_20190222"

def getFeatureMap(json_str):
    category_id = 1
    feature_map = {}
    
    images = json_str['images']
    annotations = json_str['annotations']

    for obj in annotations:
        id = obj['image_id']
        num_id = obj['category_id']
    
        if num_id is not category_id:      # 只取出类别为0的图像
            continue
            
        x, y, w, h = obj['bbox']
        feature_map[id] = ["", x, y, w, h]      # file, x, y, width, height
    
    for image in images:    # 取出id和图像文件的对应关系，加入到特征结构feature_map中
        id = image['id']
        image_file = image['file_name']
        
        if id in feature_map.keys():
            feature_map[id][0] = image_file
    
    return feature_map
        
    

if __name__ == '__main__':
    filename = dir_name + "/train_no_poly.json"
    
    with open(filename, 'r') as fd:
        line = fd.readlines()[0]
    
    json_str = json.loads(line)

    feature_map = getFeatureMap(json_str)
    # print(feature_map)
    
    ci1_map = {"images":[]}
    
    import shutil
    for id, info in feature_map.items():
        obj = {}
        obj["id"] = id
        obj["file_name"] = info[0]
        obj["x"] = info[1]
        obj["y"] = info[2]
        obj["w"] = info[3]
        obj["h"] = info[4]
        ci1_map["images"].append(obj)
        shutil.copy(dir_name + "/restricted/" + info[0], "../../datas/category_id_1/")
        
    j_str = json.dumps(ci1_map)
    print(str(j_str))
    
    with open("../../datas/category_id_1.json", "w+") as fd:
        fd.write(str(j_str))