import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 
                        kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels, 
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x += identity   # 由于输入输出feature map大小均不变，变的只是通道数，因此不需要进行下采样
        
        return x

class DarkNet53(nn.Module):
    def __init__(self, in_channels):
        super(DarkNet53, self).__init__()
        # 第一层卷积层需要保持输入输出feature map大小不变，因此需要padding
        self.mid_channels = 32
        self.conv0 = nn.Conv2d(in_channels, self.mid_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.mid_channels)  
        self.relu0 = nn.LeakyReLU(0.1) 
        self.layer1 = self._make_layer(num_layers=1, in_channels=self.mid_channels)
        self.layer2 = self._make_layer(num_layers=2, in_channels=2*self.mid_channels)
        self.layer3 = self._make_layer(num_layers=8, in_channels=4*self.mid_channels)
        self.layer4 = self._make_layer(num_layers=8, in_channels=8*self.mid_channels)
        self.layer5 = self._make_layer(num_layers=4, in_channels=16*self.mid_channels)
    
    def _make_layer(self, num_layers, in_channels):
        layers = []
        in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, 
                                        stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2*in_channels),
            nn.LeakyReLU(0.1)
        )
        layers.append(in_conv)
        
        for i in range(num_layers):
            layers.append(Block(2*in_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out_52 = x.clone()
        x = self.layer4(x)
        out_26 = x.clone()
        out_13 = self.layer5(x)
        
        return out_52, out_26, out_13


class YoloNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(YoloNet, self).__init__()
        self.num_classes = num_classes
        self.out_size = (5 + num_classes) * 3     # o,x,y,w,h,classes，这里种类需要二进制
        self.backbone = DarkNet53(in_channels)
        self.mid_channels = [1024, 768, 512, 384, 256, 128]
        self.conv_set1 = self._conv_set(self.mid_channels[0], self.mid_channels[2])
        self.output_13 = self._output_layers(self.mid_channels[2], self.out_size)
        self.conv_upsample1 = self._conv_upsample(self.mid_channels[2])
        self.conv_set2 = self._conv_set(self.mid_channels[1], self.mid_channels[4])
        self.output_26 = self._output_layers(self.mid_channels[4], self.out_size)
        self.conv_upsample2 = self._conv_upsample(self.mid_channels[4])
        self.conv_set3 = self._conv_set(self.mid_channels[3], self.mid_channels[5])
        self.output_52 = self._output_layers(self.mid_channels[5], self.out_size)

    def _conv_set(self, in_channels, out_channels):
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(out_channels, out_channels*2, kernel_size=3, 
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels*2))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(out_channels*2, out_channels, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(out_channels, out_channels*2, kernel_size=3, 
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels*2))
        layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Conv2d(out_channels*2, out_channels, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.1))

        return nn.Sequential(*layers)
    
    def _output_layers(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels*2, kernel_size=3, 
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(in_channels*2))
        layers.append(nn.LeakyReLU(0.1))                        
        layers.append(nn.Conv2d(in_channels*2, out_channels, 1, 1))   # 最后一层经过卷积层后直接输出

        return nn.Sequential(*layers)
    
    def _conv_upsample(self, in_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels//2, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(in_channels//2))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        return nn.Sequential(*layers)

    def reshape_permute(self, x, input):
        return input.reshape(x.shape[0], 3, self.num_classes+5, 
                        x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

    def forward(self, x):
        out_52, out_26, out_13 = self.backbone(x)
        x = self.conv_set1(out_13)
        out_13_final = self.output_13(x)
        out_13_final = self.reshape_permute(x, out_13_final)
        x = self.conv_upsample1(x)
        x = torch.cat([x, out_26], 1)
        x = self.conv_set2(x)
        out_26_final = self.output_26(x)
        out_26_final = self.reshape_permute(x, out_26_final)
        x = self.conv_upsample2(x)
        x = torch.cat([x, out_52], 1)
        x = self.conv_set3(x)
        out_52_final = self.output_52(x)
        out_52_final = self.reshape_permute(x, out_52_final)

        return [out_13_final, out_26_final, out_52_final]

# if __name__ == "__main__":
#     DEVICE = torch.device("cuda")
#     model = YoloNet(3, 120).to(DEVICE)
#     input = torch.zeros([8, 3, 416, 416])
#     input = input.to(DEVICE)
#     [out_13_final, out_26_final, out_52_final] = model(input)
#     print(out_13_final.shape, out_26_final.shape, out_52_final.shape)
#     print(model)
    
        
        
        
        

