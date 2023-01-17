import torch
from torchvision.models import vgg16_bn

class Classfier(torch.nn.Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.backbone = vgg16_bn(pretrained=True) # 加载在ImageNet数据集训练的VGG16模型
        self.backbone.classifier._modules['6'] = torch.nn.Linear(4096, 5) # torch.nn.Linear(4096, 1024)
        
    
    def forward(self,x):
        yp = self.backbone(x)
        return yp
        