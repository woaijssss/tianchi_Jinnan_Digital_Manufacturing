
import json
import src.submit as submit

if __name__ == '__main__':
    with open("../src/submit.json", "r+") as fd:
        lines = fd.read()
        json_str = json.loads(lines)
        results = json_str['results']
        
        imagenames = []
        for obj in results:
            imagenames.append(obj['filename'])

        dir_name = "../../jinnan2_round1_test_a_20190306/"
        submit = submit.Submit()
        image_lst = submit.getAllPictures(dir_name)[0]  # 获取测试集目录下所有的图片文件名
        
        print(len(imagenames))
        print(len(image_lst))
        
        for image in image_lst:
            if image not in imagenames:
                print(image)