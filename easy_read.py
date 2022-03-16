import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageGrab
import numpy as np
from config import *
from xml.dom import minidom

annotation_dir = "/home/MyServer/data/dogs/annotation/annotation"
imgs_dir = "/home/MyServer/data/dogs/images/images"
names_file = "/home/MyServer/My_Code/MachineLearning/dog_detect/dogs_names"

def make_names(dirs, names_file):
    '''
    ### 用于制作种类名称文件的函数
    #### dirs: 种类文件夹的路径
    #### names_file: 名称文件的保存路径
    '''
    dogs_names = os.listdir(dirs)
    f = open(names_file, 'w')
    for dog_name in dogs_names:
        f.write(dog_name)
        f.write('\n')
    f.close()

def read_names(names_file):
    '''
    ### 用于读取种类名称文件的函数
    #### names_file: 名称文件路径
    '''
    f = open(names_file, 'r')
    lines = f.readlines()
    dog_names = []
    for dog_name in lines:
        dog_name = dog_name.strip('\n')
        dog_names.append(dog_name)  
    f.close()

    return dog_names

def get_objs_info(annotation_dir):
    objs_info = []
    dom = minidom.parse(annotation_dir)     # 
    root = dom.documentElement              # 定位到根目录
    objs = root.getElementsByTagName("object")
    
    for obj in objs:
        obj_info = []
        xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
        ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
        xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
        ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
        obj_info.append(xmin)
        obj_info.append(ymin)
        obj_info.append(xmax)
        obj_info.append(ymax)
        objs_info.append(obj_info)

    return objs_info

def read_information(imgs_dir, annotations_dir, names_file):
    '''
    ### 用于提前读取文件夹内文件路径以及标签文件信息的函数, 
    ### 方便数据集的制作。
    #### imgs_dir: 图片存储路径
    #### annotations_dir: 标签文件存储路径
    #### names_file: 种类名称文件存储路径
    '''
    # 得到所有不同种狗的文件夹的名字列表这一步必须要在文本中读取提前
    # 保存好的标签名称，否则容易因为排序问题而打错标签       
    dogs_names = read_names(names_file)             
    imgs_annotations = []                           # 用于存储所有的图像路径、名称、以及标签信息
    for idx, dog_dir in enumerate(dogs_names):      # 遍历所有类型的狗
        imgs_dirs = os.path.join(imgs_dir, dog_dir) # 得到该类狗的图片文件夹路径
        annotations_dirs = os.path.join(annotations_dir, dog_dir)   # 得到该类狗的标签文件夹路径
        dog_dirs = os.listdir(annotations_dirs)     # 得到该类文件夹的所有狗标签名称
        for dog_ in dog_dirs:                       
            img_dir = os.path.join(imgs_dirs, dog_+".jpg")  # 得到此张图片的路径
            annotation_dir = os.path.join(annotations_dirs, dog_)   # 得到每个标签的路径
            img_annotation = []                     # 用于此张图片的所有信息
            img_annotation.append(img_dir)          # 保存此张图片路径
            # img_annotation.append(dog_dir)          # 保存此张图片的狗种类名称
            img_annotation.append(idx)              # 种类的标签                    
            objs_info = get_objs_info(annotation_dir)   # 存储此张图片的所有标签信息
            img_annotation.append(objs_info)
            imgs_annotations.append(img_annotation)

    return np.array(imgs_annotations, dtype=object)

if __name__ == "__main__":
    imgs_annotations = read_information(imgs_dir, annotation_dir, names_file)  
    annotations = imgs_annotations[:, 2]
    print(annotations)
    # make_names(imgs_dir, names_file)
    # names = read_names(names_file)
    # print(len(names))