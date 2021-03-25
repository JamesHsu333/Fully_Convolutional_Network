import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision
vgg = torchvision.models.vgg16(pretrained=True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = vgg
    def forward(self, inputs):
        maxpooling_map = []
        for _, layer in enumerate(vgg):
            inputs = layer(inputs)
            if isinstance(layer,nn.MaxPool2d):
                maxpooling_map.append(inputs)
        return maxpooling_map

class UpSampling(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(UpSampling,self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,2,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, inputs):
        tmp = self.model(inputs)
        return tmp

class FCN(nn.Module):
    def __init__(self, n_classes):
        super(FCN, self).__init__()
        self.backbone = VGG()
        self.up1 = UpSampling(512,512)
        self.up2 = UpSampling(512,256)
        self.up3 = UpSampling(256,128)
        self.up4 = UpSampling(128,64)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64,32,2,stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,n_classes,1)
        )

    def forward(self, inputs):
        feature_map = self.backbone(inputs)

        score = self.up1(feature_map[-1])
        score = score + feature_map[-2]
        score = self.up2(score)
        score = score + feature_map[-3]
        score = self.up3(score)
        score = score + feature_map[-4]
        score = self.up4(score)
        score = score + feature_map[-5]
        score = self.up5(score)
        return score

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

class ConfusionMatrix:
    def __init__(self, outputs, labels, class_nums):
        self.outputs = outputs
        self.labels = labels
        self.class_nums = class_nums
    def construct(self):
        self.outputs = self.outputs.flatten()
        self.outputs_count = np.bincount(self.outputs, minlength=self.class_nums)
        self.labels = self.labels.flatten()
        self.labels_count = np.bincount(self.labels, minlength=self.class_nums)

        tmp = self.labels * self.class_nums + self.outputs

        self.cm = np.bincount(tmp, minlength=self.class_nums*self.class_nums)
        self.cm = self.cm.reshape((self.class_nums, self.class_nums))

        self.Nr = np.diag(self.cm)
        self.Dr = self.outputs_count + self.labels_count - self.Nr
    def mIOU(self):
        iou = self.Nr / self.Dr
        miou = np.nanmean(iou)
        return miou

def mIOU(outputs, labels, class_nums):
    for index, (output, label) in enumerate(zip(outputs, labels)):
        output = output.transpose(1,2,0)
        output = np.argmax(output, axis=2)
        cm = ConfusionMatrix(output, label, class_nums)
        cm.construct()
        return cm.mIOU()

metrics = {
    'mIOU': mIOU,
}

if __name__ == '__main__':
    model = FCN(20+1).cuda()
    summary(model, (3, 224, 224))