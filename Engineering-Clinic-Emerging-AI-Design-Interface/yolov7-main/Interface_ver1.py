import gradio as gr
import argparse
import sys
import os
import ffmpeg
from PIL import Image
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import numpy as np
import cv2

from ourDetect import detect, generate_feature_maps
import torch

from utils.general import strip_optimizer

def correct_video(video):
    os.system("ffmpeg -i {file_str} -y -vcodec libx264 -acodec aac {file_str}.mp4".format(file_str = video))
    return video+".mp4"

def run_image(image, src, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms):
    obj_conf_thr = float(obj_conf_thr)
    iou_thr = float(iou_thr)
    agnostic_nms = bool(agnostic_nms)
    if src == "Webcam":
        image.save('Temp.jpg')  # Convert PIL image to OpenCV format if needed
        image = 'Temp.jpg'
    random = Image.open(image)
    new_dir = generate_feature_maps(random, conv_layor)
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
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                save_dir = detect(opt)
                strip_optimizer(opt.weights)
        else:
            save_dir = detect(opt)
    return [save_dir, new_dir]

def run_video(video, src, inf_size, obj_conf_thr, iou_thr, agnostic_nms):
    obj_conf_thr = float(obj_conf_thr)
    iou_thr = float(iou_thr)
    agnostic_nms = bool(agnostic_nms)
    if src == "Webcam":
        video = correct_video(video)
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
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                save_dir = detect(opt)
                strip_optimizer(opt.weights)
        else:
            save_dir = detect(opt)
    return save_dir

with gr.Blocks(title="YOLO7 Interface") as demo:
    gr.Markdown(
    """
    # Image & Video Interface for YOLO7 Model
    Upload your own image or video and watch YOLO7 try to guess what it is!
    """)
    with gr.Row():
        file_type = gr.Radio(label="File Type",info="Choose 'Image' for images, Choose 'Video' for videos",
                             choices=['Image','Video'], value='Image')
        source_type = gr.Radio(label="Source Type",
                             choices=['Computer','Webcam'], value='Computer')
        conv_layor = gr.Slider(label="Convolution Layor", minimum=1, maximum=17, value=1, interactive=True, step=1)
    with gr.Row(visible=False) as vid_tot_row:
        with gr.Row(visible=False) as vid_web_row:
            vid_web_input = gr.Video(label="Input Video",source="webcam",show_share_button=True,interactive=True)
        with gr.Row() as vid_com_row:
            vid_com_input = gr.Video(source="upload",label="Input Video",show_share_button=True,interactive=True)
        vid_output = gr.Video(label="Output Video",show_share_button=True)
    with gr.Row() as im_tot_row:
        with gr.Row(visible=False) as im_web_row:
            im_web_input = gr.Image(type='pil',source="webcam",label="Input Image")
        with gr.Row() as im_com_row:
            im_com_input = gr.Image(source="upload",type='filepath',label="Input Image",show_download_button=True,show_share_button=True,interactive=True)
        im_output = gr.Image(type='filepath',label="Output Image",show_download_button=True,show_share_button=True,interactive=False)
        im_conv_output = gr.Image(type='filepath',label="Output Convolution",show_download_button=True,show_share_button=True,interactive=False)
    with gr.Row(visible=False) as vid_tot_start:
        with gr.Row(visible=False) as vid_web_start:
            vid_web_but = gr.Button(label="Start")
            gr.ClearButton(components=[vid_web_input, vid_output],
                   interactive=True, visible=True)
        with gr.Row(visible=False) as vid_com_start:
            vid_com_but = gr.Button(label="Start")
            gr.ClearButton(components=[vid_com_input, vid_output],
                    interactive=True, visible=True)
    with gr.Row() as im_tot_start:
        with gr.Row(visible=False) as im_web_start:
            im_web_but = gr.Button(label="Start")
            gr.ClearButton(components=[im_web_input, im_output, im_conv_output],
                    interactive=True, visible=True)
        with gr.Row() as im_com_start:
            im_com_but = gr.Button(label="Start")
            gr.ClearButton(components=[im_com_input, im_output, im_conv_output],
                    interactive=True, visible=True)
    with gr.Row() as settings:
        inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0)
        obj_conf_thr = gr.Number(label='Object Confidence Threshold',value=0.25)
        iou_thr = gr.Number(label='IOU threshold for NMS',value=0.45) 
        agnostic_nms = gr.Checkbox(label='Agnostic NMS',value=True)
    def change_file_type(file, source):
        if file == "Image":
            if source == "Computer":
                return {
                        im_tot_row: gr.Row.update(visible=True),
                        vid_tot_row: gr.Row.update(visible=False),
                        im_tot_start: gr.Row.update(visible=True),
                        vid_tot_start: gr.Row.update(visible=False),
                        vid_com_row: gr.Row.update(visible=False),
                        vid_web_row: gr.Row.update(visible=False),
                        im_com_row: gr.Row.update(visible=True),
                        im_web_row: gr.Row.update(visible=False),
                        vid_web_start: gr.Row.update(visible=False),
                        vid_com_start: gr.Row.update(visible=False),
                        im_web_start: gr.Row.update(visible=False),
                        im_com_start: gr.Row.update(visible=True),
                        conv_layor: gr.Dropdown.update(visible=True)
                }
            else:
                return {
                        im_tot_row: gr.Row.update(visible=True),
                        vid_tot_row: gr.Row.update(visible=False),
                        im_tot_start: gr.Row.update(visible=True),
                        vid_tot_start: gr.Row.update(visible=False),
                        vid_com_row: gr.Row.update(visible=False),
                        vid_web_row: gr.Row.update(visible=False),
                        im_com_row: gr.Row.update(visible=False),
                        im_web_row: gr.Row.update(visible=True),
                        vid_web_start: gr.Row.update(visible=False),
                        vid_com_start: gr.Row.update(visible=False),
                        im_web_start: gr.Row.update(visible=True),
                        im_com_start: gr.Row.update(visible=False),
                        conv_layor: gr.Dropdown.update(visible=True)
                }
        else:
            if source == "Computer":
                return {
                        im_tot_row: gr.Row.update(visible=False),
                        vid_tot_row: gr.Row.update(visible=True),
                        im_tot_start: gr.Row.update(visible=False),
                        vid_tot_start: gr.Row.update(visible=True),
                        vid_com_row: gr.Row.update(visible=True),
                        vid_web_row: gr.Row.update(visible=False),
                        im_com_row: gr.Row.update(visible=False),
                        im_web_row: gr.Row.update(visible=False),
                        vid_web_start: gr.Row.update(visible=False),
                        vid_com_start: gr.Row.update(visible=True),
                        im_web_start: gr.Row.update(visible=False),
                        im_com_start: gr.Row.update(visible=False),
                        conv_layor: gr.Dropdown.update(visible=False)
                }
            else:
                return {
                        im_tot_row: gr.Row.update(visible=False),
                        vid_tot_row: gr.Row.update(visible=True),
                        im_tot_start: gr.Row.update(visible=False),
                        vid_tot_start: gr.Row.update(visible=True),
                        vid_com_row: gr.Row.update(visible=False),
                        vid_web_row: gr.Row.update(visible=True),
                        im_com_row: gr.Row.update(visible=False),
                        im_web_row: gr.Row.update(visible=False),
                        vid_web_start: gr.Row.update(visible=True),
                        vid_com_start: gr.Row.update(visible=False),
                        im_web_start: gr.Row.update(visible=False),
                        im_com_start: gr.Row.update(visible=False),
                        conv_layor: gr.Dropdown.update(visible=False)
                }
    def change_conv_layor(layor):
        return "runs\\detect\\exp\\layors\\layor" + str(int(int(layor) - 1)) + '.jpg'
    file_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor])
    source_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor])
    im_com_but.click(run_image, inputs=[im_com_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms], outputs=[im_output, im_conv_output])
    vid_com_but.click(run_video, inputs=[vid_com_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms], outputs=[vid_output])
    im_web_but.click(run_image, inputs=[im_web_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms], outputs=[im_output, im_conv_output])
    vid_web_but.click(run_video, inputs=[vid_web_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms], outputs=[vid_output])
    vid_com_input.upload(correct_video, inputs=[vid_com_input], outputs=[vid_com_input])
    vid_web_input.upload(correct_video, inputs=[vid_web_input], outputs=[vid_web_input])
    conv_layor.input(change_conv_layor, conv_layor, im_conv_output)
    demo.load()

if __name__== "__main__" :
    demo.queue().launch() 