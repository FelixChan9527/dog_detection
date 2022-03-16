from torch.utils.data import Dataset
from config import *
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
from utils import *
import easy_read

img_path = "/home/MyServer/data/dogs/images/images/n02112018-Pomeranian/n02112018_938.jpg"
save_path = "/home/MyServer/My_Code/MachineLearning/dog_detect/prediction.jpg"
def prediction(img_path, names_file, save_path):
    names = easy_read.read_names(names_file)
    box_format="midpoint"

    model = YoloNet(3, 120).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    transforms = A.Compose([
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], 
                                max_pixel_value=255,),   # 标准化，归一化
                    ToTensorV2()        # 转为tensor
                ])
    img = Image.open(img_path).convert("RGB").resize((416, 416), Image.BICUBIC)
    a = ImageDraw.ImageDraw(img)
    img_arry = np.array(img)
    augmentations = transforms(image=img_arry)
    img_arry = augmentations["image"]
    img_arry = img_arry.unsqueeze(0)
    img_arry = img_arry.to(DEVICE)
    all_pred_boxes = []
    with torch.no_grad():
        predictions = model(img_arry)
        bboxes = [[] for _ in range(1)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchors = torch.tensor([*ANCHORS[i]]).to(DEVICE) * S     # 正常大小的anchor box
            anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
            box_predictions = predictions[i][..., 1:5]
            box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
            box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
            scores = torch.sigmoid(predictions[i][..., 0:1])
            best_class = torch.argmax(predictions[i][..., 5:], dim=-1).unsqueeze(-1)
            cell_indices = (torch.arange(S)
            .repeat(predictions[i].shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions[i].device)
            )
            x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
            y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
            w_h = 1 / S * box_predictions[..., 2:4]
            converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(1, 3 * S * S, 6)
            pre_boxes = converted_bboxes.tolist()
            for idx, (box) in enumerate(pre_boxes):
                bboxes[idx] += box
            for idx in range(1):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=0.5,
                    threshold=0.15,
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
            dog_name = names[int(dog_class)]
            print(dog_name)
            a.rectangle(((xmin, ymin), (xmax, ymax)), fill=None, outline='red', width=5)
            # a.text([xmin, ymin], dog_name+" "+str(score), (255, 0, 0))
        img.save(save_path)        
                

if __name__ == "__main__":
    prediction(img_path, NAMES_FILE, save_path)
