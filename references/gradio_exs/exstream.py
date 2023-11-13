import gradio as gr
import numpy as np

def flip(im):
    return np.flipud(im)

demo = gr.Interface(
    flip, 
    gr.Image(source='webcam', streaming=True), 
    "image",
    live=True
)
demo.launch()