import gradio as gr
import argparse
import sys
import os
from PIL import Image
sys.path.append('Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/yolov7-main')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import numpy as np

from ourDetect import detect, generate_feature_maps
# from detect import detect
import torch

from utils.general import strip_optimizer

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

def run_stream(image, src, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, outputNum, is_stream):
    if is_stream:
        return run_image(image, src, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, outputNum, is_stream)
    else:
        pass
    

def run_image(image, src, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, outputNum, is_stream):
    """
    Takes an image (from upload or webcam), and outputs the yolo7 boxed output and the convolution layers

    Args:
        image (str/PIL): The file path or PIL of the the input image.
        src (str): The source of the input image, either upload or webcam
        inf_size (int): The size of the inference
        obj_conf_thr (float): The object confidence threshold
        iou_thr (float): The intersection of union number
        conv_layor (int): The number of the convolutional layer to show
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
            save_dir, smooth_dir, labels = detect(opt, outputNum=outputNum, is_stream=is_stream)
            strip_optimizer(opt.weights)
    else:
        save_dir, smooth_dir, labels = detect(opt, outputNum=outputNum, is_stream=is_stream)
    if is_stream:
        return save_dir
    return [save_dir, new_dir, smooth_dir, labels]  # added info

def run_video(video, src, inf_size, obj_conf_thr, iou_thr, agnostic_nms, is_stream, outputNum=1):
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
        else:
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
                save_dir = detect(opt, is_stream=True, outputNum=outputNum)
                strip_optimizer(opt.weights)
        else:
            save_dir = detect(opt, is_stream=True, outputNum=outputNum)
    return save_dir


# Gradio Interface Code
with gr.Blocks(title="YOLO7 Interface",theme=gr.themes.Base()) as demo:
    gr.Markdown(
    """
    # Image & Video Interface for YOLO7 Model
    Upload your own image or video and watch YOLO7 try to guess what it is!
    """)
    with gr.Row():
        file_type = gr.Radio(label="File Type",info="Choose 'Image' if you are uploading an image, Choose 'Video' if you are uploading a video",
                             choices=['Image','Video'],value='Image',show_label=True,interactive=True)
        source_type = gr.Radio(label="Source Type",info="Choose 'Computer' if you are uploading from your computer, Choose 'Webcam' if you would like to use your webcam",
                             choices=['Computer','Webcam'],value='Computer',show_label=True,interactive=True)
        conv_layor = gr.Slider(label="Convolution Layer",info="Choose a whole number from 1 to 17 to see the corresponding convolutional layer",
                               minimum=1,maximum=17,value=1,interactive=True,step=1,show_label=True)
        video_stream = gr.Checkbox(label="Stream from webcam?",info="Check this box if you would like to stream from your webcam",value=False,show_label=True,interactive=True,visible=False)
        output_map = gr.Slider(label="Map Output Number",info="Choose a whole number from 1 to 3 to see the corresponding attribution map",
                               minimum=1,maximum=3,value=1,interactive=True,step=1,show_label=True)
    with gr.Row(visible=False) as vid_tot_row:
        with gr.Row(visible=False) as vid_web_row:
            vid_web_input = gr.Video(label="Input Video",source="webcam",show_share_button=True,interactive=True)
            vid_streaming = gr.Image(type='pil',source="webcam",label="Input Image",streaming=False,visible=False,interactive=True)
        with gr.Row() as vid_com_row:
            vid_com_input = gr.Video(source="upload",label="Input Video",show_share_button=True,interactive=True)
        vid_output = gr.Video(label="Output Video",show_share_button=True)
    with gr.Row() as im_tot_row:
        with gr.Row(visible=False) as im_web_row:
            im_web_input = gr.Image(type='pil',source="webcam",label="Input Image")
        with gr.Row() as im_com_row:
            im_com_input = gr.Image(source="upload",type='filepath',label="Input Image",show_download_button=True,show_share_button=True,interactive=True)
        with gr.Row() as im_out_row:
            im_output = gr.Image(type='filepath',label="Output Image",show_download_button=True,show_share_button=True,interactive=False, visible=True)
            im_conv_output = gr.Image(type='filepath',label="Output Convolution",show_download_button=True,show_share_button=True,interactive=False)
            im_smooth_output = gr.Image(type='filepath',label="Output Smooth Gradient",show_download_button=True,show_share_button=True,interactive=False)
            labels = gr.Textbox(label='Top Predictions', value = "")   #ME
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
            gr.ClearButton(components=[im_web_input, im_output, im_conv_output, im_smooth_output, labels],  #Added info 
                    interactive=True, visible=True)
        with gr.Row() as im_com_start:
            im_com_but = gr.Button(label="Start")
            gr.ClearButton(components=[im_com_input, im_output, im_conv_output, im_smooth_output, labels],  #Added info
                    interactive=True, visible=True)
    with gr.Row() as settings:
        inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0)
        obj_conf_thr = gr.Number(label='Object Confidence Threshold',value=0.25)
        iou_thr = gr.Number(label='IOU threshold for NMS',value=0.45) 
        agnostic_nms = gr.Checkbox(label='Agnostic NMS',value=True)
        
    def change_file_type(file, source, is_stream):
        """
        Changes the visible components of the gradio interface

        Args:
            file (str): Type of the file (image or video)
            source (str): If the file is uploaded or from webcam

        Returns:
            Dictionary: Each component of the interface that needs to be updated.
        """
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
                        conv_layor: gr.Slider(visible=True),
                        video_stream: gr.Checkbox(visible=False, value=False),
                        vid_streaming: gr.Image(visible=False, streaming=False),
                        vid_web_input: gr.Video(visible=True),
                        im_out_row: gr.Row.update(visible=True),
                        im_conv_output: gr.Image(visible=True),
                        im_smooth_output: gr.Image(visible=True),
                        vid_output: gr.Video(visible=False),
                        output_map: gr.Slider(visible=True)
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
                        conv_layor: gr.Slider(visible=True),
                        video_stream: gr.Checkbox(visible=False, value=False),
                        vid_streaming: gr.Image(visible=False, streaming=False),
                        vid_web_input: gr.Video(visible=True),
                        im_out_row: gr.Row.update(visible=True),
                        im_conv_output: gr.Image(visible=True),
                        im_smooth_output: gr.Image(visible=True),
                        vid_output: gr.Video(visible=False),
                        output_map: gr.Slider(visible=True)
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
                        conv_layor: gr.Slider(visible=False),
                        video_stream: gr.Checkbox(visible=False, value=False),
                        vid_streaming: gr.Image(visible=False, streaming=False),
                        vid_web_input: gr.Video(visible=True),
                        im_out_row: gr.Row.update(visible=False),
                        im_conv_output: gr.Image(visible=True),
                        im_smooth_output: gr.Image(visible=True),
                        vid_output: gr.Video(visible=True),
                        output_map: gr.Slider(visible=False)
                }
            else:
                if is_stream:
                    return {
                            im_tot_row: gr.Row.update(visible=True),
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
                            conv_layor: gr.Slider(visible=False),
                            video_stream: gr.Checkbox(visible=True),
                            vid_streaming: gr.Image(source='webcam', visible=True, streaming=True),
                            vid_web_input: gr.Video(visible=False),
                            im_out_row: gr.Row.update(visible=True),
                            im_conv_output: gr.Image(visible=False),
                            im_smooth_output: gr.Image(visible=False),
                            vid_output: gr.Video(visible=False),
                            output_map: gr.Slider(visible=False)
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
                            conv_layor: gr.Slider(visible=False),
                            video_stream: gr.Checkbox(visible=True),
                            vid_streaming: gr.Image(visible=False, streaming=False),
                            vid_web_input: gr.Video(visible=True),
                            im_out_row: gr.Row.update(visible=False),
                            im_conv_output: gr.Image(visible=True),
                            im_smooth_output: gr.Image(visible=True),
                            vid_output: gr.Video(visible=True),
                            output_map: gr.Slider(visible=False)
                    }
                    

    
    def change_conv_layor(layor):
        """
        Changes the shown convolutional output layer based on gradio slider

        Args:
            layor (int): The layor to show

        Returns:
            str: The file path of the output image
        """
        return "outputs\\runs\\detect\\exp\\layors\\layor" + str(int(int(layor) - 1)) + '.jpg'
    
    def change_output_num(number):
        return "outputs\\runs\\detect\\exp\\smoothGrad" + str(int(int(number) -1)) + '.jpg'
    
    file_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    source_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    video_stream.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    im_com_but.click(run_image, inputs=[im_com_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, output_map, video_stream], outputs=[im_output, im_conv_output, im_smooth_output, labels])
    vid_com_but.click(run_video, inputs=[vid_com_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms, video_stream], outputs=[vid_output])
    im_web_but.click(run_image, inputs=[im_web_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, output_map, video_stream], outputs=[im_output, im_conv_output, im_smooth_output, labels])
    vid_web_but.click(run_video, inputs=[vid_web_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms, video_stream], outputs=[vid_output])
    vid_com_input.upload(correct_video, inputs=[vid_com_input], outputs=[vid_com_input])
    vid_web_input.upload(correct_video, inputs=[vid_web_input], outputs=[vid_web_input])
    conv_layor.input(change_conv_layor, conv_layor, im_conv_output)
    vid_streaming.stream(run_stream, inputs=[vid_streaming, source_type, inf_size, obj_conf_thr, iou_thr, conv_layor, agnostic_nms, output_map, video_stream], outputs=[im_output])
    output_map.input(change_output_num, output_map, im_smooth_output)
    demo.load(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layor, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map] )

if __name__== "__main__" :
    if True:
        demo.queue().launch() 
    else:
        # run_image("inference\\images\\bus.jpg","Computer",640,0.45,0.25,1,True)
        run_video("0", "Webcam", 640, 0.25, 0.45, True, True)