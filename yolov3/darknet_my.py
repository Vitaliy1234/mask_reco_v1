from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

from yolov3.utils import predict_transform


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = [int(layer) for layer in module['layers']]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                # do transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if write == 0:  # if no collector has been initialized
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weight_file):
        with open(weight_file, 'rb') as hfile:
            header = np.fromfile(hfile, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(hfile, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']

            # we load weights only when module type is convolutional
            if module_type == 'convolutional':
                model = self.module_list[i]

                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # get the number of weights of batch norm layer
                    num_bn_biases = bn.bias.numel()
                    # load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)

                conv.weight.data.copy_(conv_weights)


def parse_cfg(cfg_file):
    """
    The Function takes cfg file and returns a list of blocks.
    Each blocks describes a block in the neural network to be built.
    Block is represented as a dictionary in the list
    :param cfg_file:
    :return block_list:
    """
    with open(cfg_file, 'r') as hfile:
        cfg_list = hfile.read().split('\n')  # put all cfg lines in list

    cfg_list = [x for x in cfg_list if len(x) > 0]  # only non empty lines
    cfg_list = [x for x in cfg_list if x[0] != '#']  # without comments
    cfg_list = [x.rstrip().lstrip() for x in cfg_list]  # without fringe whitespaces

    block = {}
    blocks = []

    for line in cfg_list:
        if line[0] == '[':  # this marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    In this function we create pytorch modules of yolo net
    :param blocks:
    :return:
    """
    net_info = blocks[0]  # there is the info about our net
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        if block['type'] == 'convolutional':
            # get the info about the layer
            activation = block['activation']

            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batchNorm_{0}'.format(index), bn)

            # Check the activation
            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activ)

        # if it is an upsampling layer
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        elif block['type'] == 'route':
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])

            try:
                end = int(block['layers'][1])
            except:
                end = 0

            if start > 0:
                start -= index
            if end > 0:
                end -= index

            route = EmptyLayer()
            module.add_module('route={0}'.format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut = skip connection
        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(pos) for pos in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # take only anchors, indexed by mask
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


def get_test_input():
    img = cv2.imread('test.jpeg')
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


if __name__ == '__main__':
    model = Darknet('cfg/yolov3.cfg')
    model.load_weights('yolov3.weights')
    # inp = get_test_input()
    # pred = model.forward(inp, torch.cuda.is_available())
    # print(pred)

