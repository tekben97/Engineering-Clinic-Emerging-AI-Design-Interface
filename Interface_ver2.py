## Interface Version 2: The current interface

import sys          # System package for path dependencies
sys.path.append('Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/yolov7-main')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from run_methods import run_all, correct_video

import gradio as gr # Gradio package for interface

# Gradio Interface Code
with gr.Blocks(title="yolov7 Interface",theme=gr.themes.Base()) as demo:
    gr.Markdown(
    """
    # Image & Video Interface for yolov7 Model
    Upload your own image or video and watch yolov7 try to guess what it is!
    """)
    # Row for for input & output settings
    with gr.Row() as file_settings:
        # Allows choice for uploading image or video [for all]
        file_type = gr.Radio(label="File Type",info="Choose 'Image' if you are uploading an image, Choose 'Video' if you are uploading a video",
                             choices=['Image','Video'],value='Image',show_label=True,interactive=True,visible=True)
        # Allows choice of source, from computer or webcam [for all]
        source_type = gr.Radio(label="Source Type",info="Choose 'Computer' if you are uploading from your computer, Choose 'Webcam' if you would like to use your webcam",
                             choices=['Computer','Webcam'],value='Computer',show_label=True,interactive=True,visible=True)
        # Allows choice of which convolutional layer to show (1-17) [only for images]
        conv_layer = gr.Slider(label="Convolution Layer",info="Choose a whole number from 1 to 17 to see the corresponding convolutional layer",
                               minimum=1,maximum=17,value=1,interactive=True,step=1,show_label=True)
        # Allows choice if video from webcam is streaming or uploaded [only for webcam videos]
        video_stream = gr.Checkbox(label="Stream from webcam?",info="Check this box if you would like to stream from your webcam",value=False,show_label=True,interactive=True,visible=False)
        # Allows choice of which smooth gradient output to show (1-3) [only for images]
        output_map = gr.Slider(label="Map Output Number",info="Choose a whole number from 1 to 3 to see the corresponding attribution map",
                               minimum=1,maximum=3,value=1,interactive=True,step=1,show_label=True)
    # Row for all inputs & outputs
    with gr.Row() as inputs_outputs:
        # Default input image: Visible, Upload from computer
        input_im = gr.Image(source="upload",type='filepath',label="Input Image",
                            show_download_button=True,show_share_button=True,interactive=True,visible=True)
        # Default Boxed output image: Visible
        output_box_im = gr.Image(type='filepath',label="Output Image",
                             show_download_button=True,show_share_button=True,interactive=False,visible=True)
        # Defualt Convolutional output image: Visible
        output_conv_im = gr.Image(type='filepath',label="Output Convolution",
                                  show_download_button=True,show_share_button=True,interactive=False,visible=True)
        # Default Gradient output image: Visible
        output_grad_im = gr.Image(type='filepath',label="Output Smooth Gradient",
                                  show_download_button=True,show_share_button=True,interactive=False,visible=True)
        # Default label output textbox: Visible
        labels = gr.Textbox(label='Top Predictions:', value = "")
        # Default plaus output textbox: Visible
        plaus = gr.Textbox(label = "Plausibility Score:", value="")
        # Default time output textbox: Visible
        formatted_time = gr.Textbox(label = 'Total and Detection Time in Seconds:', value = "")
        
        # Default input video: Not visible, Upload from computer
        input_vid =  gr.Video(source="upload",label="Input Video",
                              show_share_button=True,interactive=True,visible=False)
        # Default Boxed output video: Not visible
        output_box_vid = gr.Video(label="Output Video",show_share_button=True,visible=False)
    
    # List of components for clearing
    clear_comp_list = [input_im, output_box_im, output_conv_im, output_grad_im, labels, plaus, formatted_time, input_vid, output_box_vid]
    
    # Row for start & clear buttons
    with gr.Row() as buttons:
        start_but = gr.Button(label="Start")
        clear_but = gr.ClearButton(value='Clear All',components=clear_comp_list,
                   interactive=True,visible=True)
        
    # Row for model settings
    with gr.Row() as model_settings:
        # Pixel size of the inference [Possibly useless, may remove]
        inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0)
        # Object confidence threshold
        obj_conf_thr = gr.Number(label='Object Confidence Threshold',value=0.25)
        # Intersection of union threshold
        iou_thr = gr.Number(label='IOU threshold for NMS',value=0.45)
        # Agnostic NMS boolean
        agnostic_nms = gr.Checkbox(label='Agnostic NMS',value=True)
        # Normailze gradient boolean
        norm = gr.Checkbox(label='Normalize Gradient',value=False,visible=True)
        # Weights File Upload
        weights = gr.File(label='Weights File',type='file',file_count='single',file_types=["pt"],value="weights/yolov7.pt")   
    
    def change_file_type(file, source, is_stream):
        """
        Changes the visible components of the gradio interface

        Args:
            file (str): Type of the file (image or video)
            source (str): If the file is uploaded or from webcam
            is_stream (bool): If the video is streaming or uploaded

        Returns:
            Dictionary: Each component of the interface that needs to be updated.
        """
        if file == "Image":
            if source == "Computer":
                return {
                    conv_layer: gr.Slider(visible=True),
                    video_stream: gr.Checkbox(visible=False, value=False),
                    output_map: gr.Slider(visible=True),
                    input_im: gr.Image(source="upload",type='filepath',label="Input Image",
                            show_download_button=True,show_share_button=True,interactive=True,visible=True,streaming=False),
                    output_box_im: gr.Image(visible=True),
                    output_conv_im: gr.Image(visible=True),
                    output_grad_im: gr.Image(visible=True),
                    input_vid: gr.Video(visible=False),
                    output_box_vid: gr.Video(visible=False),
                    norm: gr.Checkbox(visible=True),
                    labels: gr.Textbox(visible=True),
                    plaus: gr.Textbox(visible=True),
                    formatted_time: gr.Textbox(visible=True)
                }
            elif source == "Webcam":
                return {
                    conv_layer: gr.Slider(visible=True),
                    video_stream: gr.Checkbox(visible=False, value=False),
                    output_map: gr.Slider(visible=True),
                    input_im: gr.Image(type='pil',source="webcam",label="Input Image",
                                       visible=True,interactive=True,streaming=False),
                    output_box_im: gr.Image(visible=True),
                    output_conv_im: gr.Image(visible=True),
                    output_grad_im: gr.Image(visible=True),
                    input_vid: gr.Video(visible=False),
                    output_box_vid: gr.Video(visible=False),
                    norm: gr.Checkbox(visible=True),
                    labels: gr.Textbox(visible=True),
                    plaus: gr.Textbox(visible=True),
                    formatted_time: gr.Textbox(visible=True)
                }
        elif file == "Video":
            if source == "Computer":
                return {
                    conv_layer: gr.Slider(visible=False),
                    video_stream: gr.Checkbox(visible=False, value=False),
                    output_map: gr.Slider(visible=False),
                    input_im: gr.Image(visible=False,streaming=False),
                    output_box_im: gr.Image(visible=False),
                    output_conv_im: gr.Image(visible=False),
                    output_grad_im: gr.Image(visible=False),
                    input_vid: gr.Video(source="upload",label="Input Video",
                              show_share_button=True,interactive=True,visible=True),
                    output_box_vid: gr.Video(label="Output Video",show_share_button=True,visible=True),
                    norm: gr.Checkbox(visible=False),
                    labels: gr.Textbox(visible=False),
                    plaus: gr.Textbox(visible=True),
                    formatted_time: gr.Textbox(visible=False)
                }
            elif source == "Webcam":
                if is_stream:
                    return {
                        conv_layer: gr.Slider(visible=False),
                        video_stream: gr.Checkbox(visible=True),
                        output_map: gr.Slider(visible=False),
                        input_im: gr.Image(type='pil',source="webcam",label="Input Image",
                                           streaming=True,visible=True,interactive=True),
                        output_box_im: gr.Image(visible=True),
                        output_conv_im: gr.Image(visible=False),
                        output_grad_im: gr.Image(visible=False),
                        input_vid: gr.Video(visible=False),
                        output_box_vid: gr.Video(visible=False),
                        norm: gr.Checkbox(visible=False),
                        labels: gr.Textbox(visible=False),
                        plaus: gr.Textbox(visible=True),
                        formatted_time: gr.Textbox(visible=False)
                    }
                elif not is_stream:
                    return {
                        conv_layer: gr.Slider(visible=False),
                        video_stream: gr.Checkbox(visible=True, value=False),
                        output_map: gr.Slider(visible=False),
                        input_im: gr.Image(visible=False,streaming=False),
                        output_box_im: gr.Image(visible=False),
                        output_conv_im: gr.Image(visible=False),
                        output_grad_im: gr.Image(visible=False),
                        input_vid: gr.Video(label="Input Video",source="webcam",
                                            show_share_button=True,interactive=True,visible=True),
                        output_box_vid: gr.Video(label="Output Video",show_share_button=True,visible=True),
                        norm: gr.Checkbox(visible=False),
                        labels: gr.Textbox(visible=False),
                        plaus: gr.Textbox(visible=True),
                        formatted_time: gr.Textbox(visible=False)
                    }
                
    def change_conv_layer(layer):
        """
        Changes the shown convolutional output layer based on gradio slider

        Args:
            layer (int): The layer to show

        Returns:
            str: The file path of the output image
        """
        return "outputs\\runs\\detect\\exp\\layers\\layer" + str(int(int(layer) - 1)) + '.jpg'
    
    def change_output_num(number):
        """
        Changes the shown gradient map based on gradio slider

        Args:
            number (int): The gradient map to show

        Returns:
            str: The file path of the output image
        """
        return "outputs\\runs\\detect\\exp\\smoothGrad" + str(int(int(number) -1)) + '.jpg'
    
    # List of gradio components that change during method "change_file_type"
    change_comp_list = [conv_layer, video_stream, output_map, 
                        input_im, output_box_im, output_conv_im, output_grad_im,
                        input_vid, output_box_vid, norm, labels, plaus, formatted_time]
    # List of gradio components that are input into the run_all method (when start button is clicked)
    run_inputs = [file_type, input_im, input_vid, source_type, 
                  inf_size, obj_conf_thr, iou_thr, conv_layer, 
                  agnostic_nms, output_map, video_stream, norm, weights]
    
    # List of gradio components that are output from the run_all method (when start button is clicked)
    run_outputs = [output_box_im, output_conv_im, output_grad_im, labels, plaus, formatted_time, output_box_vid]
    
    # When these settings are changed, the change_file_type method is called
    file_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=change_comp_list)
    source_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=change_comp_list)
    video_stream.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=change_comp_list)
    # When start button is clicked, the run_all method is called
    start_but.click(run_all, inputs=run_inputs, outputs=run_outputs)
    # When video is uploaded, the correct_video method is called
    input_vid.upload(correct_video, inputs=[input_vid], outputs=[input_vid])
    # When the convolutional layer setting is changed, the change_conv_layer method is called
    conv_layer.input(change_conv_layer, conv_layer, output_conv_im)
    # When the stream setting is true, run the stream
    input_im.stream(run_all, inputs=run_inputs, outputs=run_outputs)
    # When the gradient number is changed, the change_output_num method is called
    output_map.input(change_output_num, output_map, output_grad_im)
    # When the demo is first started, run the change_file_type method to ensure default settings
    demo.load(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=change_comp_list)

#Used for debugging
if __name__== "__main__" :
    # If True, it launches Gradio interface
    # If False, it runs without the interface (for better debugging)
    if True:
        # demo.queue().launch(share=True)
        demo.queue().launch()
    else:
        #Test with yolo (weights and normal photo)
        run_all('Image','references\\inference\\images\\bus.jpg', 'None', 'Computer', 640,0.25,0.45,1,True,1,False,False,"yolov7.pt")
        #Test drone (weights and drone photo)
        #run_all('Image','references\\inference\\images\\V_DRONE_113_34.jpg', 'None', 'Computer', 640,0.25,0.45,1,True,1,False,False,"drone_63.pt")