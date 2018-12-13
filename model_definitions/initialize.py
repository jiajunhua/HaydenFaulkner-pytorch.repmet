import torch.nn as nn
from torchvision import models

from model_definitions.others.encoder import Encoder


def initialize_model(config, model_name, use_pretrained=True, use_dml=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    input_size = 0
    output_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        freeze_params(model)
        output_size = model.fc.in_features
        
        model.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)

        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        freeze_params(model)
        output_size = model.classifier[6].in_features
        
        # model.classifier[6] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
        model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        output_size = model.classifier[6].in_features
        
        # model.classifier[6] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
        model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        freeze_params(model)
        output_size = 512
        
        # model.classifier[1] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)
        model.num_classes = 1024 # num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        freeze_params(model)
        output_size = model.classifier.in_features
        
        model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        freeze_params(model)
        # Handle the auxilary net
        output_size = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
        # Handle the primary net
        output_size = model.fc.in_features
        model.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size, output_size


def freeze_params(model, params=None, verbose=True):
    for name, child in model.named_children():
        for param in child.parameters():
            if params is None or param in params:
                param.requires_grad = False
                if verbose:
                    print(name, " was frozen")
            freeze_params(child)

def _test():
    model = initialize_model(None, "resnet", use_pretrained=True, use_dml=True)
    print(model)

if __name__ == '__main__':
    _test()
