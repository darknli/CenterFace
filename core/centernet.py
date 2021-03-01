from torch import nn
from models_zoo.resnet50 import resnet50, resnet50_Decoder, resnet50_Head


class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes=2, pretrain = False):
        super(CenterNet_Resnet50, self).__init__()
        # 512,512,3 -> 16,16,2048
        self.encoder = resnet50(pretrain=pretrain)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x