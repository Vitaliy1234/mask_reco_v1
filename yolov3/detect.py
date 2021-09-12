from __future__ import division

import argparse
import torch
from yolov3.utils import load_classes, prep_image, write_results
from yolov3.darknet_my import Darknet
import time
from os.path import join, realpath
from os import listdir
from pathlib import Path
import cv2
from torch.autograd import Variable


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. "
                             "Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


def main(images, batch_size, confidence, nms_thesh, reso, weight_file, cfg_file, det_folder):
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes("data/coco.names")

    yolo, input_dim = load_network(cfg_file, weight_file, reso)

    if CUDA:
        yolo.cuda()

    yolo.eval()

    read_dir_time = time.time()
    images_list = read_input(images)

    load_batch_time = time.time()
    im_batches, im_dim_list, loaded_imgs = load_batches(batch_size, images_list, CUDA, input_dim)

    output = detection_loop(im_batches, CUDA, yolo, confidence, num_classes, nms_thesh, images_list, batch_size, classes)

    return output, loaded_imgs


def load_batches(batch_size, im_list, cuda_on, inp_dim):
    """
    Form batches from images
    :param batch_size:
    :param im_list: list of images
    :param cuda_on:
    :param inp_dim:
    :return:
    """
    loaded_imgs = [cv2.imread(img) for img in im_list]
    img_batches = list(map(prep_image, loaded_imgs, [inp_dim for _ in range(len(im_list))]))
    im_dim_list = [(image.shape[1], image.shape[0]) for image in loaded_imgs]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if cuda_on:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(im_list) // batch_size + leftover
        img_batches = [torch.cat((img_batches[i * batch_size: min((i + 1) * batch_size,
                       len(img_batches))])) for i in range(num_batches)]

    return img_batches, im_dim_list, loaded_imgs


def detection_loop(im_batches,
                   CUDA,
                   model,
                   confidence,
                   num_classes,
                   nms_thresh,
                   im_list,
                   batch_size,
                   classes):
    write = 0

    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()

    try:
        return output
    except NameError:
        print("No detections were made")
        exit()


def transform_output(im_dim_list, output, input_dim):
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(input_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (input_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (input_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    return output


def load_network(cfg, weights, reso):
    """
    Function loads yolov3 network by cfg file and weights
    :param cfg:
    :param weights:
    :param reso:
    :return:
    """
    print("Loading network.....")
    model = Darknet(cfg)
    model.load_weights(weights)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    return model, inp_dim


def read_input(img_dir):
    try:
        im_list = [join(realpath('.'), img_dir, img) for img in listdir(img_dir)]
        return im_list
    except NotADirectoryError:
        im_list = [join(realpath('.'), img_dir)]
        return im_list
    except FileNotFoundError:
        print('File or directory {} not found'.format(img_dir))
        exit()


def read_args_and_start():
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    reso = args.reso
    weight_file = args.weightsfile
    cfg_file = args.cfgfile
    det_folder = args.det

    # create dir to save results
    Path(det_folder).mkdir(exist_ok=True)

    main(images, batch_size, confidence, nms_thesh, reso, weight_file, cfg_file, det_folder)


if __name__ == '__main__':
    read_args_and_start()
