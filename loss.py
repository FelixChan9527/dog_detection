import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()             # 均方误差损失函数
        self.bce = nn.BCEWithLogitsLoss()   # 自带logic函数的二分类交叉熵损失函数
        self.entropy = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.sigmoid = nn.Sigmoid()         

        # 相关的倍增系数
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
    
    def iou_count(self, y_pre, y):     # 用于计算存在物体的位置的预测box与实际的所有box之间的iou
        xmin_ypre = y_pre[..., 0:1] - y_pre[..., 2:3] / 2
        xmax_ypre = y_pre[..., 0:1] + y_pre[..., 2:3] / 2
        ymin_ypre = y_pre[..., 1:2] - y_pre[..., 3:4] / 2
        ymax_ypre = y_pre[..., 1:2] + y_pre[..., 3:4] / 2

        xmin_y = y[..., 0:1] - y[..., 2:3] / 2
        xmax_y = y[..., 0:1] + y[..., 2:3] / 2
        ymin_y = y[..., 1:2] - y[..., 3:4] / 2
        ymax_y = y[..., 1:2] + y[..., 3:4] / 2

        x1 = torch.max(xmin_ypre, xmin_y)
        y1 = torch.max(ymin_ypre, ymin_y)
        x2 = torch.min(xmax_ypre, xmax_y)
        y2 = torch.min(ymax_ypre, ymax_y)

        i = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)     # 使其限制在最小不小于0
        s_ypre = abs((xmax_ypre - xmin_ypre) * (ymax_ypre - ymin_ypre))
        s_y = abs((xmax_y - xmin_y) * (ymax_y - ymin_y))

        iou = i / (s_ypre+s_y-i+1e-6)       # 为了避免出现分母为0的情况，加上一个很小的数

        return iou

    def forward(self, y_pre, y, anchors):      # 此处计算的损失函数为某个尺度下的预测与实际标签之间的差值
        obj = y[..., 0] == 1        # 直接取出标签中的位置来计算
        noobj = y[..., 0] == 0

        noobj_loss = self.bce(y_pre[..., 0:1][noobj], y[..., 0:1][noobj])   # 计算无物体损失

        anchors = anchors.reshape(1, 3, 1, 1, 2)    # 设置成可以与标签或预测值相计算的形式
        # 将预测的tx,ty经过sigmoid后变成网格偏移值，即与标签一样的值
        # 即计算的是网格中的相对坐标（归一化），即x、y
        xy_offset = self.sigmoid(y_pre[..., 1:3])  
        # 这个wh是实际意义上的wh，即乘以图像宽高即为真实的wh
        wh_box = torch.exp(y_pre[..., 3:5] * anchors)
        box_ypre = torch.cat([xy_offset, wh_box], dim=-1)
        box_y = y[..., 1: 5][obj]

        iou = self.iou_count(box_ypre[obj], box_y).detach()     
        # 计算每个存在物体的位置的iou，这个就是实际的物体存在置信度
        obj_loss = self.bce(y_pre[..., 0:1][obj], iou*y[..., 0:1][obj])

        y_pre[..., 1:3] = xy_offset     # x、y而不是tx、ty
        y[..., 3:5] = torch.log((1e-16 + y[..., 3:5] / anchors))    # 实际的wh变为twth
        xywh_loss = self.mse(y_pre[..., 1:5][obj], y[..., 1:5][obj])

        # torch的交叉熵函数输入的预测值为onehot形式，标签为具体值（整数）
        class_loss = self.entropy((y_pre[..., 5:][obj]), (y[..., 5][obj].long()))

        loss = (self.lambda_obj * obj_loss +
                self.lambda_noobj * noobj_loss +
                self.lambda_box * xywh_loss +
                self.lambda_class * class_loss)
        
        return loss

