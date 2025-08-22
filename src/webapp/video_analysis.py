import os
import sys
import uuid
from glob import glob

import ffmpeg
import gradio as gr

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ABS_PATH)
from geospatialib.algorithms.supervision.tracker import SuperVideoTracker

model_instance = None
gradio_inputs = None


def video_analysis_app_preview():
    print(f"Cleaning up old files... ({ABS_PATH})")
    for file in glob(f"{ABS_PATH}/annotations-grouped-*.txt"):
        os.remove(file)

    for file in glob(f"{ABS_PATH}/annotations-*.txt"):
        os.remove(file)

    for file in glob(f"{ABS_PATH}/coded-video-*.mp4"):
        os.remove(file)

    for file in glob(f"{ABS_PATH}/*_output.mp4"):
        os.remove(file)

    global gradio_inputs
    gradio_inputs = [
        gr.Dropdown(choices=get_local_models(), label="Algorithm",
                    value=get_local_models()[0], info="Select the algorithm to use"),
        gr.Slider(0.32, 0.98, value=0.35, label="Detection threshold"),
        gr.Checkbox(label="Objects Tracer enabled", value=True),
        gr.Checkbox(label="Support OBB Analyzer", value=False),
        gr.Dropdown(multiselect=True, info="Select multiple objects to detect in video",
                    choices=["car", "people", "trucks", "bus", "animals"], label="Tracking Objects",
                    value=["car", "people", "trucks", "bus"]),
        gr.Video(label='Video', format='mp4', mirror_webcam=False, sources=['upload', 'webcam']),
    ]
    outputs = [
        gr.PlayableVideo(label="Output Video")
    ]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                refresh_models_button = gr.Button("Refresh Models")
                for input_instance in gradio_inputs:
                    input_instance.render()
            with gr.Column(scale=2):
                outputs[0].render()
                start_button = gr.Button("Run Analysis")

        refresh_models_button.click(refresh_model_list, [], [gradio_inputs[0]])
        start_button.click(predict, gradio_inputs, outputs[0])
        return demo


def load_supervision_model():
    global model_instance
    if model_instance is None:
        model_instance = SuperVideoTracker()
    else:
        print("Model already loaded")


def get_local_models():
    files = glob(f"{ABS_PATH}/models/*.pt")
    if not files:
        raise FileNotFoundError("No models found in the models directory")
    return [os.path.basename(file).split(".")[0] for file in files]


def refresh_model_list():
    global gradio_inputs
    gradio_inputs[0].choices = get_local_models()


def predict(model_type, threshold, objects_tracker, obb_analyzer, objects_selected, video_path, progress=gr.Progress()):
    progress(0, desc="Starting...")

    file_name = os.path.basename(video_path).split(".")[0]

    analyzed_file_output = "./" + file_name + "_output.mp4"
    model_instance.prepare_video_file(
        objects_tracker,
        video_path,
        obb_analyzer,
        threshold,
        progress=progress,
        model_type=model_type)

    annotations = model_instance.start_file_dispatcher(video_path, analyzed_file_output)

    progress(0.99, desc="Encoding video...")

    file_guid = uuid.uuid4()
    file_output = f"./coded-video-{file_guid}.mp4"
    (
        ffmpeg
        .input(analyzed_file_output)
        .output(file_output, vcodec='libx264', crf=20, pix_fmt='yuv420p')
        .run()
    )

    progress(1.0, desc="Video analysis finished, serving results...")

    print("==========================================")
    print(f"Video analysis finished, serving results... ({file_output})")
    print(f"Model used: {model_type}")
    print(f"Threshold: {threshold}")
    print(f"Objects Tracker: {objects_tracker}")
    print("==========================================")

    return file_output
