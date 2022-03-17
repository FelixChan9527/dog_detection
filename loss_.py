import torch
import torch.nn as nn

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # 相关的倍增系数
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # 直接得到是否存在物体的掩膜（妙）
        obj = target[..., 0] == 1       
        noobj = target[..., 0] == 0

        # 没有物体的损失
        no_object_loss = self.bce(
            (predictions[..., 0: 1][noobj]), (target[..., 0: 1][noobj]) 
        )

        # 存在物体的损失
        # 这个就是用来计算pw*exp(tw)的pw（先验框），
        # 这一步把某一尺度的三个anchor box变为与神经
        # 网络输出相同维度数目的形状以便输出的每一个
        # 网格的参数与之相乘
        anchors = anchors.reshape(1, 3, 1, 1, 2)    
        # 得到pw*exp(tw)后的值，其中x, y属于0~1, w, h可能大于1
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), 
                    torch.exp(predictions[..., 3:5] * anchors)], dim=-1)
        # 计算对应位置的iou
        # .detach()是为了保持其梯度，为了计算更加有保证，
        # 去掉可能不一定会出现问题，此处作者也不敢确定
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # 此处的target与iou相乘，即选取实际存在物体的预测框和
        # 真实框之间的iou作为目标置信度，其中iou以及提前乘以obj
        # 我觉得这个乘以target[..., 0:1]是多余的（@@@）
        object_loss = self.bce((predictions[..., 0:1][obj]), 
                                    (ious * target[..., 0:1][obj]))
        
        # 坐标的损失
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors))   # 得到真实的tw, th
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # 种类损失
        # 很奇怪，非得有5:
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()))
        
        # print(box_loss.item(), no_object_loss.item(), object_loss.item(), class_loss.item())
        
        return (
            self.lambda_box * box_loss + 
            self.lambda_noobj * no_object_loss +
            self.lambda_obj * object_loss + 
            self.lambda_class * class_loss
            )
        
