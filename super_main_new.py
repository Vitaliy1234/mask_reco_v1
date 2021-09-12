import argparse
import os
import random
import time
import torch
from numpy import argmax
from torchvision import transforms
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
from yolov3.utils import load_classes
from yolov3.detect import load_network, load_batches, detection_loop, transform_output


N_IMAGES = 250  # n images loaded per iter
FACE_CONFIDENCE = 0.99  # MTCNN face confidence
SAVE_TXT = False


def launch():
    # parse arguments for yolo
    args = parse_arguments()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    reso = args.reso
    weight_file = args.weightsfile
    cfg_file = args.cfgfile
    det_folder = args.det
    n_samples = args.n_samples

    # load yolo
    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes("data/coco.names")

    yolo, input_dim = load_network(cfg_file, weight_file, reso)

    if CUDA:
        yolo.cuda()
    yolo.eval()

    print('Loading employee/client classifier...')
    model_ft = torch.load('model_empl_cl_upd_upd.pth')
    model_ft.eval()
    print('Classifier loaded!')
    print()
    print('Loading MTCNN')
    face_detector = MTCNN(device=torch.device('cuda'))
    face_detector.eval()
    print('MTCNN loaded')
    print()
    print('Loading Mask classifier...')
    mask_classifier = torch.load('model_mask.pth')
    mask_classifier.eval()
    print('Mask classifier loaded!')

    if CUDA:
        model_ft.cuda()
        mask_classifier.cuda()
    # create list of file names with images
    list_of_img_files = load_img_files(n_samples, images)
    # point start time
    start_all = time.time()
    # var for counting errors
    count_incorrects = 0
    number_of_no_mask = 0

    data_transforms_cl_1 = transform_image(256, 224)  # transform image for employee-client classifier
    data_transforms_cl_2 = transform_image(256, 224)  # transform image for mask-no_mask classifier

    for i in range(0, len(list_of_img_files), N_IMAGES):
        start_iter = time.time()
        # launch yolo
        cur_list_ims = list_of_img_files[i:i + N_IMAGES]
        try:
            im_batches, im_dim_list, loaded_imgs = load_batches(batch_size, cur_list_ims, CUDA, input_dim)
        except:
            continue
        output = detection_loop(im_batches, CUDA, yolo, confidence, num_classes, nms_thesh, cur_list_ims, batch_size,
                                classes)
        output = transform_output(im_dim_list, output, input_dim)
        persons_only = [pers for pers in find_persons(output)]
        imgs_rb = [swapRB(img) for img in loaded_imgs]
        # output = transform_output(im_dim_list, output, input_dim)
        print('==================================')
        print('YOLO done!')
        print('Starting Employee-client classifier...')
        # launch employee/client classifier
        employees = empl_cl_classifier(model_ft, persons_only, imgs_rb, data_transforms_cl_1)
        print('==================================')
        print('Employee-client classifier done!')
        print('Starting face detection...')
        # launch face detection model
        empls_faces = start_mtcnn(face_detector, employees, imgs_rb)
        print('==================================')
        print('Face detection done!')
        print('Starting mask classifier')
        employee_detections, face_b_boxes = start_mask_classifier(mask_classifier, empls_faces, imgs_rb,
                                                                  data_transforms_cl_2)
        print('==================================')
        print('Mask classifier done')
        print()
        print('Drawing bounding boxes and saving results...')
        draw_b_boxes_and_save(employee_detections, face_b_boxes, loaded_imgs, list_of_img_files, det_folder, images,
                              SAVE_TXT)
        number_of_no_mask += len(face_b_boxes)
        print('Results has been written! Time:', time.time() - start_iter)
        print('Iter_num:', i // N_IMAGES, 'of', len(list_of_img_files) // N_IMAGES)
        print('FPS:', N_IMAGES // (time.time() - start_iter))
        print('No mask count:', number_of_no_mask)

        del loaded_imgs
        del imgs_rb

    print('Time:', time.time() - start_all)
    print('Incorrects: ', count_incorrects)
    print('Number of no mask:', number_of_no_mask)


def find_persons(output_yolo):
    for output in output_yolo:
        if int(output[-1]) == 0:
            yield output


def transform_image(param_size: int, param_crop: int):
    return transforms.Compose([transforms.Resize(param_size),
                               transforms.CenterCrop(param_crop),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def swapRB(img):
    """
    Xchange red and blue channels
    :param img:
    :return img_rb: pic with xchanged channels
    """
    red = img[:, :, 2].copy()
    blue = img[:, :, 0].copy()

    img_rb = img.copy()

    img_rb[:, :, 0] = red
    img_rb[:, :, 2] = blue

    return img_rb


def empl_cl_classifier(model_ft, out_yolo, ims_transformed, data_transforms):
    """
    Evaluate employee-client classifier
    :param model_ft: employee-client classifier
    :param out_yolo: people detections from yolov3
    :param ims_transformed: images transformed to pytorch format
    :param data_transforms: torch transforms for images
    :return result: list with relative (person is employee) yolo detections
    """
    result = []
    with torch.no_grad():
        for output in out_yolo:
            c1 = tuple(output[1:3].int())
            c2 = tuple(output[3:5].int())
            img = ims_transformed[int(output[0])]

            # crop person from image
            person = img[c1[1]:c2[1], c1[0]:c2[0]]
            if person.shape[0] == 0 or person.shape[1] == 0:
                continue

            person_tr = data_transforms(Image.fromarray(person)).unsqueeze(0)
            person_tr = person_tr.cuda()

            start_empl_cl = time.time()
            output_person = model_ft(person_tr)
            _, preds = torch.max(output_person, 1)
            print('Employee/client classifier took:', time.time() - start_empl_cl, preds)

            if preds == 1:  # if person is employee
                result.append(output)

    return result


def start_mtcnn(model_mtcnn, employees, ims_transformed):
    """
    Starts mtcnn for face detection
    :param model_mtcnn: mtcnn model
    :param employees: detections from yolo with only employees
    :param ims_transformed: images transformed to pytorch format
    :return result: list with tuples (detection with employee, bounding box with face)
    """
    result = []
    with torch.no_grad():
        for empl_detection in employees:
            c1 = tuple(empl_detection[1:3].int())
            c2 = tuple(empl_detection[3:5].int())
            employee = ims_transformed[int(empl_detection[0])][c1[1]:c2[1], c1[0]:c2[0]]
            start_time = time.time()
            try:
                result_face_detect = model_mtcnn.detect(employee)
            except:
                continue
            print('MTCNN took', time.time() - start_time)

            if result_face_detect[0] is not None:  # if face was found
                indx = argmax(result_face_detect[1])
                max_conf = result_face_detect[1][indx]
                b_box_max = result_face_detect[0][indx]  # bbox with max confidence

                if max_conf >= FACE_CONFIDENCE:  # filter faces through threshold
                    result.append((empl_detection, b_box_max))

    return result


def start_mask_classifier(model_mask, empl_faces, ims_transformed, data_transforms):
    """
    Starting mask classifier
    :param model_mask:
    :param empl_faces:
    :param ims_transformed:
    :return result: list of tuples (employee_detection, face_bounding_box)
    """
    result_detections = []
    result_b_boxes = []
    with torch.no_grad():
        for empl_detection, b_box_max in empl_faces:
            c1 = tuple(empl_detection[1:3].int())
            c2 = tuple(empl_detection[3:5].int())
            cur_index = int(empl_detection[0])
            employee = ims_transformed[cur_index][c1[1]:c2[1], c1[0]:c2[0]]

            x1_face, y1_face = max(int(b_box_max[0]), 0), max(int(b_box_max[1]), 0)
            x2_face, y2_face = max(int(b_box_max[2]), 0), max(int(b_box_max[3]), 0)

            face_img = employee[y1_face:y2_face, x1_face:x2_face]

            face_tr = data_transforms(Image.fromarray(face_img)).unsqueeze(0)
            face_tr = face_tr.cuda()
            start = time.time()
            output_mask = model_mask(face_tr)
            print('Mask model took:', time.time() - start)
            _, preds_mask = torch.max(output_mask, 1)

            if preds_mask == 1:  # if class is "no mask"
                result_detections.append(empl_detection)
                cur_list_b_boxes = [cur_index]
                cur_list_b_boxes.extend(b_box_max)
                result_b_boxes.append(cur_list_b_boxes)

    return result_detections, result_b_boxes


def draw_b_boxes_and_save(employee_detections, faces, loaded_ims, ims_files, det_folder, images, save_txt = False):
    """
    Draw bounding boxes on faces without masks
    :param faces:
    :param employee_detections:
    :param images:
    :param det_folder:
    :param loaded_ims:
    :param ims_files:
    :return:
    """
    for empl_detection, face in zip(employee_detections, faces):
        cur_im_index = int(empl_detection[0])
        img = loaded_ims[cur_im_index]
        c1 = tuple(empl_detection[1:3].int())
        x1_face, y1_face = max(int(face[1]), 0), max(int(face[2]), 0)
        x2_face, y2_face = max(int(face[3]), 0), max(int(face[4]), 0)
        t_size = cv2.getTextSize('No_mask', cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        # coordinates of label box
        c1_face_text = c1[0] + x1_face, c1[1] - t_size[1] + y1_face - 1
        c2_face_text = c1[0] + t_size[0] + x1_face, c1[1] + y1_face
        # coordinates for face
        face_1 = (c1[0] + x1_face, c1[1] + y1_face)
        face_2 = (c1[0] + x2_face, c1[1] + y2_face)
        # bound face
        cv2.rectangle(img, face_1, face_2, color=(0, 0, 255),
                      thickness=2)
        # draw box for label
        cv2.rectangle(img, c1_face_text, c2_face_text, color=(0, 0, 200), thickness=-1)
        # print label in box
        cv2.putText(img, 'No_mask', (c1[0] + x1_face + 1, c1[1] + y1_face - 1), cv2.FONT_HERSHEY_PLAIN,
                    2, [255, 255, 255], 2)

        # saving image
        cur_file_name = ims_files[cur_im_index].replace('/', '=')
        full_name = os.path.join(det_folder,
                                 str(cur_im_index) + cur_file_name[cur_file_name.find(images.replace('/', '=')) + len(images):])
        cv2.imwrite(full_name, img)

        if save_txt:
            with open(full_name[:full_name.rfind('.')] + '.txt', 'w') as hfile:
                print(img.shape)
                print(face_1, face_2)
                face_1 = face_1[0].cpu().numpy() / img.shape[1], face_1[1].cpu().numpy() / img.shape[0]
                face_2 = face_2[0].cpu().numpy() / img.shape[1], face_2[1].cpu().numpy() / img.shape[0]
                face_1_yolo = (face_2[0] + face_1[0]) / 2, (face_2[1] + face_2[0]) / 2
                face_2_yolo = face_2[0] - face_1[0], face_2[1] - face_1[1]
                print(face_1_yolo, face_2_yolo)
                hfile.write('0' + ' ' + str(face_1_yolo[0]) + ' ' + str(face_1_yolo[1]) + ' ' + str(face_2_yolo[0]) + ' ' + str(face_2_yolo[1]))


def load_img_files(n_samples, input_dir):
    """
    Function for creating list of file names in input dir
    :param n_samples: n files to sample from all files
    :param input_dir: dir from which we get file names
    :return: list of file names in input dir
    """
    list_of_files = []

    for os_walk_elem in os.walk(input_dir):
        for img_file in os_walk_elem[2]:
            full_img_path = os.path.join(os_walk_elem[0], img_file)
            list_of_files.append(full_img_path)

    if n_samples > 0:
        list_of_files = random.sample(list_of_files, n_samples)

    return list_of_files


def parse_arguments():
    """
    Function for command line arguments parsing
    :return: dict of parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="yolov3/cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. "
                             "Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument('-n', '--n_samples', type=int, default=0, help='Number of files to process')

    return parser.parse_args()


if __name__ == '__main__':
    launch()
