import gc
import os
import shutil
import sys
from glob import glob

import gradio as gr
import torch

from segment_tiff import gradio_app_gui

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from segment_tiff import load_model
from video_analysis import video_analysis_app_preview, load_supervision_model
from webapp.realtime_stream import video_live_stream_gui

tab_names = [
    "Segmentation",
    "Video Tracking",
    "Real-time Video Stream"
]

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Remove existing directories
    for match in glob("./masks_*"):
        shutil.rmtree(match, ignore_errors=True)

    for match in glob("./tiles_*"):
        shutil.rmtree(match, ignore_errors=True)

    # Load the model
    load_model()
    load_supervision_model()

    # CSS for the Gradio app
    css = """
    #component-20{
    background: #0175FF;
    border: unset;
    }
    
    #output_video {
        width: 100%;
        height: 820px;
    }
    
    iframe {
        width: 100%;
        height: 820px;
    }

    #component-12 {

    }

    #component-12 h1::before{
 
    }
    """

    gr.TabbedInterface([
        gradio_app_gui(),
        video_analysis_app_preview(),
        video_live_stream_gui()
    ],
        tab_names,
        css=css,
        theme=gr.themes.Default(),
        title="RIUS") \
        .queue(api_open=True) \
        .launch(server_name="0.0.0.0",
                server_port=7501,
                inline=False,
                share=False,
                show_api=True)
