import torch
import os
from PIL import Image
import argparse

import sys
sys.path.append('Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/yolov7-main')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from ourDetect import detect, generate_feature_maps # used for output generation
from utils.general import strip_optimizer # used for opt creation


def correct_video(video):
    """
    Takes a video file of any type and turns it into a gradio compatible .mp4/264 video

    Args:
        video (str): The file path of the input video

    Returns:
        str: The file path of the output video
    """
    os.system("ffmpeg -i {file_str} -y -vcodec libx264 -acodec aac {file_str}.mp4".format(file_str = video))
    return video+".mp4"

def run_all(source_type, im, vid, src, inf_size=640, obj_conf_thr=0.25, iou_thr=0.45, conv_layer=1, agnostic_nms=False, outputNum=1, is_stream=False, norm=False):
    if is_stream:
        return run_image(image=im,src=src,inf_size=inf_size,obj_conf_thr=obj_conf_thr,iou_thr=iou_thr,conv_layer=conv_layer,agnostic_nms=agnostic_nms,outputNum=outputNum,is_stream=is_stream,norm=norm)
    elif source_type == "Image":
        return run_image(image=im,src=src,inf_size=inf_size,obj_conf_thr=obj_conf_thr,iou_thr=iou_thr,conv_layer=conv_layer,agnostic_nms=agnostic_nms,outputNum=outputNum,is_stream=is_stream,norm=norm)
    elif source_type == "Video":
        return run_video(video=vid,src=src,inf_size=inf_size,obj_conf_thr=obj_conf_thr,iou_thr=iou_thr,agnostic_nms=agnostic_nms,is_stream=is_stream,outputNum=outputNum)

def run_image(image, src, inf_size, obj_conf_thr, iou_thr, conv_layer, agnostic_nms, outputNum, is_stream, norm):
    """
    Takes an image (from upload or webcam), and outputs the yolo7 boxed output and the convolution layers

    Args:
        image (str/PIL): The file path or PIL of the the input image.
        src (str): The source of the input image, either upload or webcam
        inf_size (int): The size of the inference
        obj_conf_thr (float): The object confidence threshold
        iou_thr (float): The intersection of union number
        conv_layer (int): The number of the convolutional layer to show
        agnostic_nms (bool): The agnostic nms boolean

    Returns:
        List[str]: A list of strings, where each string is a file path to an output image.
    """
    obj_conf_thr = float(obj_conf_thr)
    iou_thr = float(iou_thr)
    agnostic_nms = bool(agnostic_nms)
    if src == "Webcam":
        image.save('Temp.jpg')  # Convert PIL image to OpenCV format if needed
        image = 'Temp.jpg'
    if not is_stream:
        random = Image.open(image)
        new_dir = generate_feature_maps(random, conv_layer)
    if agnostic_nms:
        agnostic_nms = 'store_true'
    else:
        agnostic_nms = 'store_false'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=image, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=inf_size, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=obj_conf_thr, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou_thr, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action=agnostic_nms, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='outputs/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    opt.no_trace = True
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov7.pt']:
            save_dir, smooth_dir, labels, formatted_time = detect(opt, outputNum=outputNum, is_stream=is_stream, norm=norm)
            strip_optimizer(opt.weights)
    else:
        save_dir, smooth_dir, labels, formatted_time = detect(opt, outputNum=outputNum, is_stream=is_stream, norm=norm)
    if is_stream:
        return [save_dir, None, None, None, None, None]
    return [save_dir, new_dir, smooth_dir, labels, formatted_time, None]  # added info

def run_video(video, src, inf_size, obj_conf_thr, iou_thr, agnostic_nms, is_stream, outputNum=1, norm=False):
    """
    Takes a video (from upload or webcam), and outputs the yolo7 boxed output

    Args:
        video (str): The file path of the input video
        src (str): The source of the input video, either upload or webcam
        inf_size (int): The size of the inference
        obj_conf_thr (float): The object confidence threshold
        iou_thr (float): The intersection of union number
        agnostic_nms (bool): The agnostic nms boolean

    Returns:
        str: The file path of the output video
    """
    obj_conf_thr = float(obj_conf_thr)
    iou_thr = float(iou_thr)
    agnostic_nms = bool(agnostic_nms)
    if src == "Webcam":
        if is_stream:
            video = "0"
    if agnostic_nms:
        agnostic_nms = 'store_true'
    else:
        agnostic_nms = 'store_false'
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=video, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=inf_size, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=obj_conf_thr, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou_thr, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action=agnostic_nms, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='outputs/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    opt.batch_size = 1
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                save_dir = detect(opt, outputNum=outputNum, is_stream=is_stream, norm=norm)
                strip_optimizer(opt.weights)
        else:
            save_dir = detect(opt, outputNum=outputNum, is_stream=is_stream, norm=norm)
    return [None, None, None, None, None, save_dir]