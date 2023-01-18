from modeling.backbone.resnet18 import feature_resnet18, feature_resnet50

NET_OUT_DIM = {'resnet18': 512, 'resnet50': 2048}

def build_feature_extractor(backbone):
    if backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return feature_resnet18()
    elif backbone == "resnet50":
        print("Feature extractor: ResNet-50")
        return feature_resnet50()
    else:
        raise NotImplementedError