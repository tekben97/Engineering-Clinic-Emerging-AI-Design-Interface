import gradio as gr
from PIL import Image
import io
import subprocess
import os

# Define a function to handle the image input
def detect_objects(input_image):
    # Save the uploaded image temporarily inside the "inference" folder
    print(input_image)
    
    # Run your YOLOv7 detection script
    subprocess.run(["python", r"C:\Users\asien\Documents\GitHub\Engineering-Clinic-Emerging-AI-Design-Interface\yolov7-main\detect.py", "--source", input_image, "--project", "run_images", "--name", "exp"])


    # Load the output image from your detection
    output_image = Image.open("C:\\Users\\asien\\Documents\\GitHub\\Engineering-Clinic-Emerging-AI-Design-Interface\\run_images\\exp\\image.png")
    return output_image

# Define the Gradio interface with a run button
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.inputs.Image(type="filepath", source="upload"),
    outputs=gr.outputs.Image(type="pil"),
    live=False  # Set live=False to disable real-time updates
)

# Launch the Gradio interface
iface.launch(share=True)
