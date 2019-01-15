from torchvision import models

from model_definitions.others.encoder import Encoder
from model_definitions.cnns.basics.protonet import ProtoNet
from model_definitions.cnns.basics.lenet import LeNet
from model_definitions.cnns.inceptions.googlenet import GoogLeNet
from model_definitions.detectors.faster_rcnn.faster_rcnn import FasterRCNN


def initialize_model(config, model_name, model_id):

    input_size = 0
    output_size = 0
    mean = None
    std = None

    if model_name == "resnet":
        """ Resnet18
        """
        model = models.resnet18(pretrained=config.model.use_pretrained)

        # BE SURE TO NORMALISE, can do by replacing the fc layers
        if model_id == '01':
            freeze_params(model)
        elif model_id == '02':
            pass  # don't freeze

        output_size = model.fc.in_features

        model.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)
        # output_size = config.model.emb_size

        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    #
    # elif model_name == "alexnet":
    #     """ Alexnet
    #     """
    #     model = models.alexnet(pretrained=config.model.use_pretrained)
    #     if model_id == '01':
    #         freeze_params(model)
    #     elif model_id == '02':
    #         pass  # don't freeze
    #     output_size = model.classifier[6].in_features
    #
    #     # model.classifier[6] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
    #     model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)
    #     input_size = 224
    #
    # elif model_name == "vgg":
    #     """ VGG11_bn
    #     """
    #     model = models.vgg11_bn(pretrained=config.model.use_pretrained)
    #     if model_id == '01':
    #         freeze_params(model)
    #     elif model_id == '02':
    #         pass  # don't freeze
    #     output_size = model.classifier[6].in_features
    #
    #     # model.classifier[6] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Linear(output_size, num_classes)
    #     model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)
    #     input_size = 224
    #
    # elif model_name == "squeezenet":
    #     """ Squeezenet
    #     """
    #     model = models.squeezenet1_0(pretrained=config.model.use_pretrained)
    #     if model_id == '01':
    #         freeze_params(model)
    #     elif model_id == '02':
    #         pass  # don't freeze
    #     output_size = 512
    #
    #     # model.classifier[1] = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=1024)  # nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    #     model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)
    #     model.num_classes = 1024 # num_classes
    #     input_size = 224
    #
    # elif model_name == "densenet":
    #     """ Densenet
    #     """
    #     model = models.densenet121(pretrained=config.model.use_pretrained)
    #     if model_id == '01':
    #         freeze_params(model)
    #     elif model_id == '02':
    #         pass  # don't freeze
    #     output_size = model.classifier.in_features
    #
    #     model.classifier = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)  # nn.Linear(output_size, num_classes)
    #     input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=config.model.use_pretrained)
        if model_id == '01':
            freeze_params(model)
        elif model_id == '02':
            pass  # don't freeze
        # Handle the auxilary net
        output_size = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)  # nn.Linear(output_size, num_classes)
        # Handle the primary net
        output_size = model.fc.in_features
        model.fc = Encoder(input_size=output_size, hidden_sizes=[2048], output_size=config.model.emb_size)  # nn.Linear(output_size, num_classes)
        input_size = 299
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    elif model_name == "protonet":
        if model_id == '01':
            input_size = 28
            output_size = config.model.emb_size

            model = ProtoNet(x_dim=1, hid_dim=64, z_dim=output_size)

    elif model_name == "fasterRCNN":
        if model_id == '01':
            # input_size = 28
            # output_size = config.model.emb_size
            input_size = None
            output_size = None
            model = FasterRCNN(output_size=config.model.emb_size,
                               config=config)

    # elif model_name == "lenet":
    #     if model_id == '01':
    #         input_size = 28
    #         output_size = config.model.emb_size  # todo maybe alter this, put as config option
    #
    #         model = LeNet(emb_dim=output_size)
    #
    # elif model_name == "googlenet":
    #     if model_id == '01':
    #         input_size = 32
    #         output_size = config.model.emb_size  # todo maybe alter this, put as config option
    #
    #         model = GoogLeNet(output_dim=output_size)
    else:
        print("Invalid model name: %s" % model_name)
        exit()

    return model, input_size, output_size, mean, std


def freeze_params(model, params=None, verbose=True):
    for name, child in model.named_children():
        for param in child.parameters():
            if params is None or param in params:
                param.requires_grad = False
                if verbose:
                    print(name, " was frozen")  # TODO this never ends in densenet, can we put check in (either pass up changed or us timer)
            freeze_params(child)

# def _test():
    # model = initialize_model(None, "resnet")
    # print(model)

# if __name__ == '__main__':
    # _test()
