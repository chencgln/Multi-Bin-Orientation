from models.backbones import darknet, vgg

def backbone_fn(backbone):
    if backbone=="darknet53":
        return darknet.darknet53
    elif backbone=="resnet18":
        return vgg.vgg16