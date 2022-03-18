from torch.utils.data import Dataset
from config import *
from PIL import Image, ImageFont
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
import easy_read
import time

font_path = "/home/MyServer/My_Code/MachineLearning/dog_detect/FORTE.TTF"
img_path = "/home/MyServer/data/dogs/images/images/n02086240-Shih-Tzu/n02086240_1725.jpg"
save_path = "/home/MyServer/My_Code/MachineLearning/dog_detect/prediction.jpg"

class test_model():
    def __init__(self, model_path, names_file, class_num):
        self.names = easy_read.read_names(names_file)
        self.model = YoloNet(3, class_num).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transforms = A.Compose([
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1],
            max_pixel_value=255,),   # 标准化，归一化
            ToTensorV2()        # 转为tensor
            ])
    
    def iou_count(self, box1, box2):     # 用于计算存在物体的位置的预测box与实际的所有box之间的iou
        xmin1 = box1[..., 0:1] - box1[..., 2:3] / 2
        xmax1 = box1[..., 0:1] + box1[..., 2:3] / 2
        ymin1 = box1[..., 1:2] - box1[..., 3:4] / 2
        ymax1 = box1[..., 1:2] + box1[..., 3:4] / 2

        xmin2 = box2[..., 0:1] - box2[..., 2:3] / 2
        xmax2 = box2[..., 0:1] + box2[..., 2:3] / 2
        ymin2 = box2[..., 1:2] - box2[..., 3:4] / 2
        ymax2 = box2[..., 1:2] + box2[..., 3:4] / 2

        x1 = torch.max(xmin1, xmin2)
        y1 = torch.max(ymin1, ymin2)
        x2 = torch.min(xmax1, xmax2)
        y2 = torch.min(ymax1, ymax2)

        i = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)     # 使其限制在最小不小于0
        s_box1 = abs((xmax1 - xmin1) * (ymax1 - ymin1))
        s_box2 = abs((xmax2 - xmin2) * (ymax2 - ymin2))

        iou = i / (s_box1+s_box2-i+1e-6)       # 为了避免出现分母为0的情况，加上一个很小的数

        return iou

    def get_imgarray(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((416, 416), Image.BICUBIC)
        
        img_arry = np.array(img)
        augmentations = self.transforms(image=img_arry)
        img_arry = augmentations["image"]
        img_arry = img_arry.unsqueeze(0)
        img_arry = img_arry.to(DEVICE)

        return img_arry, img

    def nms(self, boxes, iou_threshold, obj_threshold):     # NMS算法
        obj_boxes = []
        obj_boxes = sorted(boxes, key=lambda x: x[1], reverse=True) # 按置信度大小进行排列

        obj_boxes = [box for box in boxes if box[1] > obj_threshold]    # 筛选出大于置信度的box并保留

        final_boxes = []
        while obj_boxes:
            chosen_box = obj_boxes.pop(0)   # 取出置信度最大的box
            obj_boxes = [box for box in obj_boxes
                                if box[0] != chosen_box[0] 
                                or self.iou_count(
                                    torch.tensor(chosen_box[2:]),
                                    torch.tensor(box[2:]),
                                    )< iou_threshold
                        ]
            final_boxes.append(chosen_box)
        return final_boxes
    
    def plot_img(self, img, boxes):
        for i in range(len(boxes)):
            dog_class, score, x, y, w, h = boxes[i]
            x = x * 416
            y = y * 416
            w = w * 416
            h = h * 416
            xmin = int(x - w//2)
            xmax = int(x + w//2)
            ymin = int(y - h//2)
            ymax = int(y + h//2)
            dog_name = self.names[int(dog_class)]
            print(dog_name, score)
            a = ImageDraw.ImageDraw(img)
            font = ImageFont.truetype(font_path, 20)
            a.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, outline='red', width=5)
            a.text((xmin, ymin-20), dog_name, fill="red", font=font)

    def prediction(self, img_path, save_path):
        img_arry, img_original = self.get_imgarray(img_path)    # 图像转换
        with torch.no_grad():       # 以下操作均在无梯度下进行，以防改变参数
            predictions = self.model(img_arry)      # 预测
            bboxes = []
            for i in range(3):      # 遍历三个尺度
                predictions[i] = predictions[i].squeeze(0)      # 把batch维度去掉
                S = predictions[i].shape[1]
                anchors = torch.tensor([*ANCHORS[i]]).to(DEVICE) * S     # 正常大小的anchor box
                anchors = anchors.reshape(len(anchors), 1, 1, 2)

                box_predictions = predictions[i][..., 1:5]
                # 将输出的tx,ty转为x,y（网格中的相对位置）
                box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
                # 将tw,th变为真实的w,h，相对于该尺度
                box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
                # 提取出置信度，表示是否存在物体
                scores = torch.sigmoid(predictions[i][..., 0:1])    # 预测的置信度
                # 取出种类。对于论文，作者是通过一个阈值来取出这个box的种类，因为它可能属于多类
                # 为了方便，这里就直接取最大种类作为该类
                best_class = torch.argmax(predictions[i][..., 5:], dim=-1).unsqueeze(-1)
                # 取出网格位置
                cell_indices = (torch.arange(S).repeat(3, S, 1).unsqueeze(-1).to(DEVICE))
                # 计算出绝对位置，并将x,y,w,h从该feature map的尺度归一化
                x = (box_predictions[..., 0:1] + cell_indices) / S
                y = (box_predictions[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) / S
                w_h = 1 / S * box_predictions[..., 2:4]
                # 将同一个尺度下所有的box（无论有无物体）的数据都放在一个list里面
                converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(3 * S * S, 6)
                pre_boxes = converted_bboxes.tolist()

                bboxes += pre_boxes     # 累积三个尺度的所有box

            # nms筛选三个尺度的所有框
            final_boxes = self.nms(bboxes, iou_threshold=0.5, obj_threshold=0.2)

            self.plot_img(img_original, final_boxes)    # 画框

            img_original.save(save_path)


if __name__ == "__main__":
    test = test_model(MODEL_PATH, NAMES_FILE, 120)
    test.prediction(img_path, save_path)
    # prediction(img_path, NAMES_FILE, save_path)
    # dataset = YoloDataset(IMGS_PATH, ANNATATIONS_PATH, NAMES_FILE, 
    #                         IMG_SIZE, ANCHORS)
    # train_loader = DataLoader(dataset, 16, shuffle=True)
    # model = YoloNet(3, 120).to(DEVICE)
    # model.load_state_dict(torch.load(MODEL_PATH))
    # model.eval()
    # check_class_accuracy(model, train_loader, 0.05)
