import gradio as gr
import argparse
import sys
import os
import ffmpeg
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from ourDetect import detect
import torch

from utils.general import strip_optimizer

def correct_video(video):
    os.system("ffmpeg -i {file_str} -y -vcodec libx264 -acodec aac {file_str}.mp4".format(file_str = video))
    return video

def run_image(image, inf_size, obj_conf_thr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=image, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=int(inf_size), help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=obj_conf_thr, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
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

def run_video(video):
    video = correct_video(video)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=video, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
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
    file_type = gr.Radio(label="File Type",info="Choose 'Image' for images, Choose 'Video' for videos",
                      choices=['Image','Video'], value='Image')
    with gr.Row(visible=False) as vid_row:
        vid_input = gr.Video(label="Input Video",show_share_button=True,interactive=True)
        vid_output = gr.Video(label="Output Video",show_share_button=True)
    with gr.Row() as im_row:
        im_input = gr.Image(type='filepath',label="Input Image",show_download_button=True,show_share_button=True,interactive=True)
        im_output = gr.Image(type='filepath',label="Output Image",show_download_button=True,show_share_button=True,interactive=False)
    with gr.Row(visible=False) as vid_start:
        vid_but = gr.Button(label="Start")
        gr.ClearButton(components=[
                   vid_input, vid_output, im_input, im_output],
                   interactive=True, visible=True)
    with gr.Row() as im_start:
        im_but = gr.Button(label="Start")
        gr.ClearButton(components=[
                   vid_input, vid_output, im_input, im_output],
                   interactive=True, visible=True)
    with gr.Row() as img_exs:
        im_ex = gr.Examples(label="Image Examples",
                    examples=['inference/images/zidane.jpg',
                            'inference/images/image1.jpg',
                            'inference/images/image2.jpg',
                            'inference/images/image3.jpg',
                            'inference/images/bus.jpg',
                            'inference/images/horses.jpg'], 
                            inputs=[im_input], 
                            outputs=[im_output], 
                            fn=run_image, 
                            cache_examples=True)
    with gr.Row(visible=False) as vid_exs:
        vid_ex = gr.Examples(label="Video Examples",
                    examples=["inference/videos/ducks.mp4",
                              "inference/videos/sample-5s.mp4"], 
                            inputs=[vid_input], 
                            outputs=[vid_output], 
                            fn=run_video, 
                            cache_examples=True)
    with gr.Row() as settings:
        inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0)
        obj_conf_thr = gr.Number(label='Object Confidence Threshold',value=0.25)
        
    def change_file_type(type):
        if type == "Image":
            return {
                im_row: gr.Row.update(visible=True),
                vid_row: gr.Row.update(visible=False),
                im_start: gr.Row.update(visible=True),
                vid_start: gr.Row.update(visible=False),
                img_exs: gr.Row.update(visible=True),
                vid_exs: gr.Row.update(visible=False)
            }
        else:
            return {
                im_row: gr.Row.update(visible=False),
                vid_row: gr.Row.update(visible=True),
                im_start: gr.Row.update(visible=False),
                vid_start: gr.Row.update(visible=True),
                img_exs: gr.Row.update(visible=False),
                vid_exs: gr.Row.update(visible=True)
            }
    file_type.input(change_file_type, show_progress=True, inputs=[file_type], outputs=[im_row, vid_row, im_start, vid_start, img_exs, vid_exs])
    im_but.click(run_image, inputs=[im_input, inf_size, obj_conf_thr], outputs=[im_output])
    vid_but.click(run_video, inputs=[vid_input], outputs=[vid_output])
    vid_input.upload(correct_video, inputs=[vid_input], outputs=[vid_input])
    demo.load()

if __name__== "__main__" :
    demo.queue().launch() 
