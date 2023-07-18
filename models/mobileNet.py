import torchvision
import torch.nn as nn

def mobilenet_v3_large(config):
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, config.classes_num)
    return model

def mobilenet_v3_small(config):
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, config.classes_num)
    return model