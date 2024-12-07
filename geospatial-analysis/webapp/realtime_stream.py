import time

import gradio as gr

from shared.rpc_client import RPCClient
from webapp.video_analysis import get_local_models

rpc_client = RPCClient('127.0.0.1')


def run_video_analysis(model_type, threshold, objects_tracker, obb_analyzer, objects_selected, video_path,
                       video_stream_path,
                       progress=gr.Progress()):
    global rpc_client
    time_waiting = 25
    progress(0, desc="Starting...")
    rpc_client.connect()
    progress(0.05, desc="Connected.")

    if video_stream_path is not None and video_stream_path != "":
        rpc_client.start_rtsp_stream(video_stream_path)
        progress(0.15, desc="RTSP Stream started.")
    else:
        rpc_client.start_analysis(video_path)
        progress(0.15, desc="RTSP input Stream started.")

    progress(0.10, desc="Analysis starting.")

    while True:
        if time_waiting >= 100:
            break
        time.sleep(1)
        progress(time_waiting / 100, desc="Analysis started, please wait for stream to load.")
        time_waiting += 3

    return """
    <iframe src="http://localhost:8888/stream" width="100%"></iframe>
    """


def video_live_stream_gui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(choices=get_local_models(), label="Algorithm",
                                    value="rius_maskrcnn_daniel")
                thresh = gr.Slider(0.32, 0.98, value=0.35, label="Detection threshold")
                tracer = gr.Checkbox(label="Objects Tracer enabled", value=True)
                obb = gr.Checkbox(label="Support OBB Analyzer", value=False)
                objects = gr.Dropdown(multiselect=True, info="Select multiple objects to detect in video",
                                      choices=["car", "people", "trucks", "bus", "animals"], label="Tracking Objects",
                                      value=["car", "people", "trucks", "bus"])
            with gr.Column(scale=2):
                video_path = gr.Video(label='Video', mirror_webcam=False, sources=['upload', 'webcam'])
                video_stream_path = gr.Textbox(label='RTSP Addr', placeholder='rtsp://', value="")
                start_button = gr.Button("Start")
        with gr.Row():
            with gr.Column(scale=2, min_width=640):
                output = gr.HTML(label="Output Video", elem_id="output_video")

        start_button.click(run_video_analysis, [
            model,
            thresh,
            tracer,
            obb,
            objects,
            video_path,
            video_stream_path
        ], output)

        return demo
