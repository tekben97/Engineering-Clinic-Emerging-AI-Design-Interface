## Interface Version 1: NOT USED ** Here for reference only **

import gradio as gr # Gradio package for interface
import sys          # System package for path dependencies
sys.path.append('Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/Interface_Dependencies')
sys.path.append('Engineering-Clinic-Emerging-AI-Design-Interface/yolov7-main')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from run_methods import run_image, run_video, correct_video #run_stream

# Gradio Interface Code
with gr.Blocks(title="YOLO7 Interface",theme=gr.themes.Base()) as demo:
    gr.Markdown(
    """
    # Image & Video Interface for YOLO7 Model
    Upload your own image or video and watch YOLO7 try to guess what it is!
    """)
    # For for input & output settings
    with gr.Row() as file_settings:
        # Allows choice for uploading image or video [for all]
        file_type = gr.Radio(label="File Type",info="Choose 'Image' if you are uploading an image, Choose 'Video' if you are uploading a video",
                             choices=['Image','Video'],value='Image',show_label=True,interactive=True)
        # Allows choice of source, from computer or webcam [for all]
        source_type = gr.Radio(label="Source Type",info="Choose 'Computer' if you are uploading from your computer, Choose 'Webcam' if you would like to use your webcam",
                             choices=['Computer','Webcam'],value='Computer',show_label=True,interactive=True)
        # Allows choice of which convolutional layer to show (1-17) [only for images]
        conv_layer = gr.Slider(label="Convolution Layer",info="Choose a whole number from 1 to 17 to see the corresponding convolutional layer",
                               minimum=1,maximum=17,value=1,interactive=True,step=1,show_label=True)
        # Allows choice if video from webcam is streaming or uploaded [only for videos]
        video_stream = gr.Checkbox(label="Stream from webcam?",info="Check this box if you would like to stream from your webcam",value=False,show_label=True,interactive=True,visible=False)
        # Allows choice of which smooth gradient output to show (1-3) [only for images]
        output_map = gr.Slider(label="Map Output Number",info="Choose a whole number from 1 to 3 to see the corresponding attribution map",
                               minimum=1,maximum=3,value=1,interactive=True,step=1,show_label=True)
    # For video inputs & outputs
    with gr.Row(visible=False) as vid_tot_row:
        # For webcam video inputs
        with gr.Row(visible=False) as vid_web_row:
            # For webcam video input
            vid_web_input = gr.Video(label="Input Video",source="webcam",show_share_button=True,interactive=True)
            # For webcam streaming input
            vid_streaming = gr.Image(type='pil',source="webcam",label="Input Image",streaming=False,visible=False,interactive=True)
        # For computer video inputs
        with gr.Row() as vid_com_row:
            # For computer video input
            vid_com_input = gr.Video(source="upload",label="Input Video",show_share_button=True,interactive=True)
        # For video output
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
            labels = gr.Textbox(label='Top Predictions:', value = "")   #ME
            formatted_time = gr.Textbox(label = 'Time to Run in Seconds:', value = "")
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
            gr.ClearButton(components=[im_web_input, im_output, im_conv_output, im_smooth_output, labels, formatted_time],  #Added info 
                    interactive=True, visible=True)
        with gr.Row() as im_com_start:
            im_com_but = gr.Button(label="Start")
            gr.ClearButton(components=[im_com_input, im_output, im_conv_output, im_smooth_output, labels, formatted_time],  #Added info
                    interactive=True, visible=True)
    with gr.Row() as settings:
        inf_size = gr.Number(label='Inference Size (pixels)',value=640,precision=0)
        obj_conf_thr = gr.Number(label='Object Confidence Threshold',value=0.25)
        iou_thr = gr.Number(label='IOU threshold for NMS',value=0.45) 
        agnostic_nms = gr.Checkbox(label='Agnostic NMS',value=True)
        norm = gr.Checkbox(label='Normalize Gradient',value=False)
        
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
                        conv_layer: gr.Slider(visible=True),
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
                        conv_layer: gr.Slider(visible=True),
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
                        conv_layer: gr.Slider(visible=False),
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
                            conv_layer: gr.Slider(visible=False),
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
                            conv_layer: gr.Slider(visible=False),
                            video_stream: gr.Checkbox(visible=True),
                            vid_streaming: gr.Image(visible=False, streaming=False),
                            vid_web_input: gr.Video(visible=True),
                            im_out_row: gr.Row.update(visible=False),
                            im_conv_output: gr.Image(visible=True),
                            im_smooth_output: gr.Image(visible=True),
                            vid_output: gr.Video(visible=True),
                            output_map: gr.Slider(visible=False)
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
        return "outputs\\runs\\detect\\exp\\smoothGrad" + str(int(int(number) -1)) + '.jpg'
    
    file_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layer, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    source_type.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layer, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    video_stream.input(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layer, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map])
    im_com_but.click(run_image, inputs=[im_com_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layer, agnostic_nms, output_map, video_stream], outputs=[im_output, im_conv_output, im_smooth_output, labels, formatted_time])
    vid_com_but.click(run_video, inputs=[vid_com_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms, video_stream], outputs=[vid_output])
    im_web_but.click(run_image, inputs=[im_web_input, source_type, inf_size, obj_conf_thr, iou_thr, conv_layer, agnostic_nms, output_map, video_stream], outputs=[im_output, im_conv_output, im_smooth_output, labels, formatted_time])
    vid_web_but.click(run_video, inputs=[vid_web_input, source_type, inf_size, obj_conf_thr, iou_thr, agnostic_nms, video_stream], outputs=[vid_output])
    vid_com_input.upload(correct_video, inputs=[vid_com_input], outputs=[vid_com_input])
    vid_web_input.upload(correct_video, inputs=[vid_web_input], outputs=[vid_web_input])
    conv_layer.input(change_conv_layer, conv_layer, im_conv_output)
    vid_streaming.stream(run_stream, inputs=[vid_streaming, source_type, inf_size, obj_conf_thr, iou_thr, conv_layer, agnostic_nms, output_map, video_stream, norm], outputs=[im_output])
    output_map.input(change_output_num, output_map, im_smooth_output)
    demo.load(change_file_type, show_progress=True, inputs=[file_type, source_type, video_stream], outputs=[im_tot_row, vid_tot_row, im_tot_start, vid_tot_start, vid_com_row, vid_web_row, im_com_row, im_web_row, vid_web_start, vid_com_start, im_web_start, im_com_start, conv_layer, video_stream, vid_streaming, vid_web_input, im_out_row, im_conv_output, im_smooth_output, vid_output, output_map] )

if __name__== "__main__" :
    # If True, it launches Gradio interface
    # If False, it runs without the interface
    if True:
        demo.queue().launch(share=True) 
    else:
        # run_image("inference\\images\\bus.jpg","Computer",640,0.45,0.25,1,True)
        run_video("0", "Webcam", 640, 0.25, 0.45, True, True)