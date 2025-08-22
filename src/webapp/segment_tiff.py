import os
import ssl
import sys
import time
import uuid
import warnings

import gradio as gr
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import shared
from algorithms.io.geotiff import GeoTiffHandler
from algorithms.visioniq.sam import SegmentAnything

print = shared.log_instance().info
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")

out_dir = os.path.join(os.path.dirname(__file__), "./")
checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

model_instance: SegmentAnything

Image.MAX_IMAGE_PIXELS = None


def gradio_app_gui():
    inputs = [
        gr.Dropdown(choices=[], label="Algorithm", value="fastSAM_GPU"),
        gr.Slider(0.1, 0.99, value=0.31, label="Box threshold"),
        gr.Slider(0.1, 0.99, value=0.31, label="Text threshold"),
        gr.File(type="filepath", label='Blob'),
        gr.Textbox(lines=1, label="Phrase query for objects", placeholder="Enter RIUS labels here"),
        gr.Textbox(lines=1, label="Center Coordinates (Optional)", placeholder="x,y", value="20,0"),
        gr.Textbox(lines=1, label="Local Path (Optional)", placeholder="file://...", value=""),
    ]
    outputs = [
        gr.Chatbot(label="RIUS"),
        gr.Image(type="pil", label="Output Image"),
        gr.File(label="GeoJSON"),
        gr.Image(type="pil", label="Output Image"),
    ]
    return gr.Interface(inputs=inputs,
                        allow_flagging="never",
                        outputs=outputs,
                        title="Retrieval Inference Utility Service",
                        fn=predict,
                        examples=[])


def predict(sam_type, box_threshold, text_threshold, tiff_path, text_prompt, center_coords, file_url):
    start_time = time.time()
    global model_instance

    session_guid = uuid.uuid4()

    shared.log_instance().info(f"Started Prediction process for session: {session_guid}.")

    chat_history = []
    shared.log_instance().info("In processing... " + text_prompt)

    original_text_prompt = text_prompt
    text_prompt = text_prompt \
        .replace("find", "") \
        .replace("count", "") \
        .replace("all", "") \
        .replace("only", "") \
        .strip()

    if file_url is not None and len(file_url) > 0:
        tiff_path = file_url

    print(tiff_path)
    file_name = os.path.basename(tiff_path)
    shared.log_instance().info(f"Processing file: {file_name}")

    # check what type of file is it (jpeg or tiff)
    if tiff_path.endswith(".jpeg") or tiff_path.endswith(".jpg") or tiff_path.endswith(".png"):
        print("JPEG file detected.")
        return_results = model_instance.predict_image(text_prompt, tiff_path)
    else:
        print("TIFF file detected.")
        geo_tiff_handler = GeoTiffHandler(tiff_path)
        return_results = model_instance.predict_tiff(text_prompt, geo_tiff_handler)

    if len(return_results) == 0:
        print("RIUS could not find any object in the image.")
        return chat_history, None, None, None, None

    output_image_tiff = model_instance.get_solution_results() + "merged.tiff"
    output_image = model_instance.get_solution_results() + "merged.png"
    output_masks = model_instance.get_solution_results() + "merged.geojson"

    if os.path.exists(output_image_tiff) is False:
        output_image_tiff = None

    if os.path.exists(output_image) is False:
        output_image = None

    if os.path.exists(output_masks) is False:
        output_masks = None

    one_label = return_results[len(return_results) - 1][2][0]

    total_logits = []
    total_phases = []
    for result in return_results:
        if result[3] is not None:
            for phrase, logit in zip(result[2], result[3].numpy().astype(float).tolist()):
                total_phases.append(phrase)
                total_logits.append(logit)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(total_phases, total_logits)]
    print(f"GEO-TIFF Predicted: {labels[0]}, total found: {len(labels)}")
    chat_history.append((original_text_prompt, "I found " + str(len(labels)) + " " + one_label + "."))
    shared.log_instance().info(f"Predicted: {chat_history}")

    print("--- running process took: %s seconds ---" % (time.time() - start_time))
    return chat_history, output_image, output_masks, output_image_tiff


def load_model():
    global model_instance
    model_instance = SegmentAnything()
    shared.log_instance().info("Model built.")
    shared.log_instance().info("App ready.")
