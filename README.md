# tianchi_Jinnan_Digital_Manufacturing
津南数字制造算法挑战赛代码

## json标注中的keys说明：  
（1）images：  
（2）annotations： 每个图像的注释（image_id对应与"images"中的id）  
```
    {
      "id": 1,              // 序号
      "image_id": 0,        // 对应于"images"中的id
      "category_id": 1,     // 对应于"categories"中的id
      "iscrowd": 0,         // 是否是人群？（待确认）
      "segmentation": [     // 初赛为空

      ],
      "area": [             // 初赛为空

      ],
      "bbox": [             // 图像中危险品大致的位置信息
        388.0,              // x
        207.0,              // y
        53.0,               // width
        56.0                // height
      ]
    }
（3）info：        该标注文件的描述性信息
（4）categories：  类别信息（类别id和对应的类别名称）
（5）licenses：
```  

## 比赛过程  
### 二分类器训练过程  
- 根据annotations中的category_id，将其中一个类别取出，单独训练  
```python
（1）将训练集目录下restricted所有category_id为0的图像取出，放入./datas/category_id_1目录中  
（2）

```
