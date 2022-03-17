from torch.utils.data import Dataset
from config import *
from easy_read import *
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch

class YoloDataset(Dataset):
    def __init__(self, imgs_dir, annotations_dir, names_file, input_size, 
                anchors, classes=120, FM_size=[13, 26, 52], iou_thresh=0.5):
        self.input_size = input_size
        self.classes = classes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        self.FM_size = FM_size
        self.iou_thresh = iou_thresh
        self.imgs_annotations = read_information(imgs_dir, annotations_dir, names_file)
        self.imgs_dirs, self.classes, self.boxes_infos = \
                            self.separate_item(self.imgs_annotations)
        self.imgs = self.read_pics(self.imgs_dirs)
        self.transforms = A.Compose([
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),   # 标准化，归一化
            ToTensorV2()        # 转为tensor
        ])
        
    def separate_item(self, imgs_annotations):      # 划分信息
        imgs_dirs = imgs_annotations[:, 0]
        classes = imgs_annotations[:, 1]
        boxes_infos = imgs_annotations[:, 2]

        return imgs_dirs, classes, boxes_infos
    
    def read_pics(self, imgs_dirs):
        '''
        ### 读取所有图片到内存
        '''
        imgs = []
        for img_dir in imgs_dirs:
            img = Image.open(img_dir).convert("RGB")
            imgs.append(img)
        return imgs
    
    def iou(self, gt_box, anchors):     # 计算一个box和9个anchor box的iou
        min_w = torch.min(gt_box[0], anchors[..., 0])
        min_h = torch.min(gt_box[1], anchors[..., 1])
        i = min_w * min_h
        gt_s = gt_box[0] * gt_box[1]          # GT的面积
        anchors_s = anchors[..., 0] * anchors[..., 1]   # anchor box的面积
        u = gt_s + anchors_s - i                        # 计算面积的并集

        return i / u

    def __len__(self):
        return len(self.imgs_annotations)
    
    def img_trans(self, img):
        '''
        ### 用于图像的修整, 包括将图像修整为416*416、图像增强等
        #### img: 输入图像
        #### new_img: 输出图像处理后归一化并转为tensor的图像数据
        #### dw、dh: 图像粘贴到416*416灰图中间的坐标
        #### scale: 原始图与标准图之间尺寸变换的比例
        '''
        long_side = max(img.size)                   # 获取长的边
        w, h = img.size
        scale = self.input_size / long_side         # 得到
        w, h = int(w*scale), int(h*scale)           # 宽高按比例缩放
        img2 = img.resize((w, h), Image.BICUBIC)    
        dw, dh = (self.input_size - w), (self.input_size - h)       # 得到宽高的差值
        new_img = Image.new('RGB', (self.input_size, self.input_size), (128,128,128))   # 得到标准尺寸的灰图，用于加灰条
        new_img.paste(img2, (dw//2, dh//2))         # 粘贴变换尺寸后的图到灰图上
        new_img = np.array(new_img)                 # 图像增强需要以narray的格式输入
        augmentations = self.transforms(image=new_img)  # 图像增强
        new_img = augmentations["image"]

        return new_img, dw, dh, scale

    def size_trans(self, img, boxes_info, dog_class):

        new_img, dw, dh, scale = self.img_trans(img)
        targets = [torch.zeros((3, S, S, 6)) for S in self.FM_size]      

        if len(boxes_info):     # 如果有ground true，则处理ground true坐标
            np.random.shuffle(boxes_info)       # 随机打乱一个图的所有ground true
            for box_info in boxes_info:
                gt_box = []
                box_info *= scale
                box_info[0] += dw//2
                box_info[2] += dw//2
                box_info[1] += dh//2
                box_info[3] += dh//2
                gt_box.append(box_info[0])
                gt_box.append(box_info[1])
                gt_box.append(box_info[2])
                gt_box.append(box_info[3])
                gt_box.append(dog_class)
                gt_box = np.array(gt_box, dtype=np.float32)
                gt_box[0: 4] /= self.input_size
                gt_box[2: 4] = gt_box[2: 4] - gt_box[0: 2]      # 得wh
                gt_box[0: 2] = gt_box[0: 2] + gt_box[2: 4] / 2  # 得xy
                
                iou = self.iou(torch.tensor(gt_box[2: 4]), self.anchors)    # 计算与9个anchor的iou
                anchor_idxes = iou.argsort(descending=True, dim=0)    # 按iou从大到小排列anchors的索引
                x, y, w, h, dog_class = gt_box      # 得到box的数据
                has_anchor = [False, False, False]

                for idx, anchor_idx in enumerate(anchor_idxes):     # 遍历所有的anchor，第一个一定为gt所在的位置  
                    
                    feature = anchor_idx // 3       # 得到对应的尺度位置
                    anchor = anchor_idx % 3         # 得到一个feature中anchor的位置
                    s = self.FM_size[feature]       # 得到对应feature map大小
                    i, j = int(s*y), int(s*x)       # 得到对应网格位置

                    # if idx == 0:                    # 即最大的iou为ground true所在的位置
                    if not has_anchor[feature]:    
                        gt_x, gt_y = s*x-j, s*y-i   # 小数部分即为偏移
                        gt_w, gt_h = s*w, s*h       # 变为对应尺度下的宽高
                        targets[feature][anchor, i, j, 0] = 1   # 存在ground true
                        targets[feature][anchor, i, j, 1:5] = torch.tensor([gt_x, gt_y, gt_w, gt_h])   # 赋值
                        targets[feature][anchor, i, j, 5] = torch.tensor(dog_class)
                    
                    elif iou[anchor_idx] > self.iou_thresh: # iou大于门限但不是最大iou, 直接忽略
                        targets[feature][anchor, i, j, 0] = -1  # 直接令其为-1即可

                    '''
                    ### 此处的obj的位置的值只是一个标志
                    ### 直接标志是否存在物体或者是否需要忽略
                    '''

        return new_img, targets
    
    def __getitem__(self, index: int):
        img = self.imgs[index]
        dog_class = self.classes[index]
        boxes_info = np.array(self.boxes_infos[index], dtype=np.float32)
        new_img, targets = self.size_trans(img, boxes_info, dog_class)
        return new_img, targets
        


