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
from utils import *
import easy_read

font_path = "/home/MyServer/My_Code/MachineLearning/dog_detect/FORTE.TTF"
img_path = "/home/MyServer/data/dogs/images/images/n02086240-Shih-Tzu/n02086240_1725.jpg"
save_path = "/home/MyServer/My_Code/MachineLearning/dog_detect/prediction.jpg"

class test_model():
    def __init__(self, model_path, names_file, class_num):
        self.names = easy_read.read_names(names_file)
        self.model = YoloNet(3, class_num).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def prediction(self, img_path, save_path):
        self.names = easy_read.read_names(names_file)    # 获取种类的名单

        transforms = A.Compose([
                        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], 
                                    max_pixel_value=255,),   # 标准化，归一化
                        ToTensorV2()        # 转为tensor
                    ])
        img = Image.open(img_path).convert("RGB").resize((416, 416), Image.BICUBIC)
        a = ImageDraw.ImageDraw(img)
        font = ImageFont.truetype(font_path, 20)
        img_arry = np.array(img)
        augmentations = transforms(image=img_arry)
        img_arry = augmentations["image"]
        img_arry = img_arry.unsqueeze(0)
        img_arry = img_arry.to(DEVICE)

        all_pred_boxes = []
        box_format="midpoint"
        with torch.no_grad():       # 以下操作均在无梯度下进行，以防改变参数
            predictions = self.model(img_arry)
            bboxes = [[] for _ in range(1)]
            for i in range(3):
                S = predictions[i].shape[2]
                anchors = torch.tensor([*ANCHORS[i]]).to(DEVICE) * S     # 正常大小的anchor box
                anchors = anchors.reshape(1, len(anchors), 1, 1, 2)

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
                # 取出网格位置@@@@
                cell_indices = (torch.arange(S)
                .repeat(predictions[i].shape[0], 3, S, 1)
                .unsqueeze(-1)
                .to(predictions[i].device)
                )
                # 计算出绝对位置，并将x,y,w,h从该feature map的尺度归一化
                x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
                y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
                w_h = 1 / S * box_predictions[..., 2:4]
                # 将同一个尺度下所有的box（无论有无物体）的数据都放在一个list里面
                converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(1, 3 * S * S, 6)
                pre_boxes = converted_bboxes.tolist()

                for idx, (box) in enumerate(pre_boxes):     # 把对应图片的所有box都放一起
                    bboxes[idx] += box                           # 三个尺度的所有box都放一起   
            
            # nms筛选三个尺度的所有框
            for idx in range(1):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=0.5,
                    threshold=0.2,
                    box_format=box_format,
                )
                for nms_box in nms_boxes:
                    all_pred_boxes.append(nms_box)
            
            for i in range(len(all_pred_boxes)):
                dog_class, score, x, y, w, h = all_pred_boxes[i]
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
                a.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, outline='red', width=5)
                a.text((xmin, ymin-20), dog_name, fill="red", font=font)

            img.save(save_path)


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
