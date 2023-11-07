import gradio as gr
import argparse
import sys
sys.path.append('./')

from ourDetect import detect
import torch
from utils.general import strip_optimizer

# Define a function to run YOLOv7 with the provided settings
def run(weights, conf_thres, iou_thres, agnostic_nms, source):
    weights = weights.strip()  # Remove any leading/trailing spaces
    conf_thres = float(conf_thres)
    iou_thres = float(iou_thres)
    agnostic_nms = bool(agnostic_nms)
    source = source.strip()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=[weights], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=conf_thres, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou_thres, help='IOU threshold for NMS')
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

    with torch.no_grad():
        if opt.update:
            for opt.weights in weights:
                save_dir = detect(opt)
                strip_optimizer(opt.weights)
        else:
            save_dir = detect(opt)
    return save_dir + "\zidane.jpg"

# Define the Gradio settings block
settings_block = [
    "text",  # "text" component for Weights (Path)
    "number",  # "number" component for Confidence Threshold
    "number",  # "number" component for IoU Threshold
    "checkbox",  # "checkbox" component for Agnostic NMS
    "text"  # "text" component for Source (Path)
]

# Create a Gradio interface for YOLOv7 settings
iface = gr.Interface(
    fn=run,
    inputs=settings_block,
    outputs="text",  # Use "text" directly as the output type
    live=True
)


if __name__ == "__main__":
    iface.launch()