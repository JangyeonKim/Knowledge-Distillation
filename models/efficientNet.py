import torchvision
import torch.nn as nn

def efficientNet_b0(config):
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model

def efficientNet_b1(config) :
    model = torchvision.models.efficientnet_b1(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model

def efficientNet_b2(config) :
    model = torchvision.models.efficientnet_b2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model
    
def efficientNet_b3(config) :
    model = torchvision.models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model

def efficientNet_b7(config):
    model = torchvision.models.efficientnet_b7(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model

def efficientNet_v2_s(config):
    model = torchvision.models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model

def efficientNet_v2_m(config):
    model = torchvision.models.efficientnet_v2_m(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.classes_num)
    return model
