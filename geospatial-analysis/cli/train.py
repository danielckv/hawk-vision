import json
import os
import time

import neptune
import pandas as pd
from ultralytics import YOLO

ROOT_DATASET_DIR = "../datasets"
CURRENT_DATASET_DIR = f"{ROOT_DATASET_DIR}/"


def generate_yolo_labels(json_filepath, image_directory=None):
    """
    Generates YOLOv5 labels from a COCO-like JSON annotation file and image paths.

    Args:
        json_filepath: Path to the COCO-like JSON annotation file.
        image_directory: (Optional) Directory containing the images.
                         Defaults to the same directory as the JSON file.

    Returns:
        None (saves label files directly to disk).
    """

    # Load JSON data
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Determine image directory if not provided
    if image_directory is None:
        image_directory = os.path.dirname(json_filepath)

    if os.path.exists(f"{image_directory}/../labels") is False:
        os.makedirs(f"{image_directory}/../labels", exist_ok=True)

    # Process each image's annotations
    for image in data["images"]:
        image_id = image["id"]
        image_filename = image["file_name"]

        labels = []  # Store labels for this image

        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                class_id = annotation["category_id"]
                x, y, width, height = annotation["bbox"]

                # Calculate YOLO format
                image_width, image_height = image["width"], image["height"]
                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                yolo_width = width / image_width
                yolo_height = height / image_height
                labels.append(f"{class_id} {x_center} {y_center} {yolo_width} {yolo_height}")

        # Write labels to a text file
        label_filename = image_filename.replace(".PNG", ".txt")
        label_filepath = os.path.join(image_directory, "../labels", label_filename)  # Use os.path.join
        with open(label_filepath, "w") as f:
            f.write("\n".join(labels))


def start_training(dataset_dir):
    global CURRENT_DATASET_DIR
    CURRENT_DATASET_DIR = dataset_dir
    # check if the dataset directory exists
    if os.path.exists(f"{CURRENT_DATASET_DIR}/annotations/instances_default.json"):
        generate_yolo_labels(f"{CURRENT_DATASET_DIR}/annotations/instances_default.json",
                             f"{CURRENT_DATASET_DIR}/images")

        # create yolo config yaml file
        with open(f"{CURRENT_DATASET_DIR}/config.yaml", "w") as f:
            f.write("train: images\n")
            f.write("val: images\n")
            f.write("nc: 5\n")
            f.write("names: ['person','car','truck','motorcycle','bus']\n")
            f.write("img_size: 640\n")
            f.write("batch_size: 8\n")
            f.write("epochs: 4\n")
            f.write("device: 'cuda'\n")

    model_name_ver = time.strftime("%Y.%m-%d.%H%M%S")
    model_name_ver = "v" + model_name_ver

    # Initialize Neptune run
    run = neptune.init_run(
        custom_run_id=model_name_ver,
        project="mobilegroup-research/RIUS-SIRIUS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjU5OGQ3Zi05OWM1LTQyYWYtODcwMy04MGZjNTc2YjM2MTcifQ==",
    )

    model_version = neptune.init_model_version(
        model="RIUS-ARLVHCL",
        project="mobilegroup-research/RIUS-SIRIUS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjU5OGQ3Zi05OWM1LTQyYWYtODcwMy04MGZjNTc2YjM2MTcifQ==",
        # your credentials
    )

    run["train/images"].track_files(f"{CURRENT_DATASET_DIR}/images")

    model = YOLO("../models/yolov8m.pt")

    # Train the model and log metrics to Neptune
    model.train(data=f"{CURRENT_DATASET_DIR}/config.yaml", project=f"{CURRENT_DATASET_DIR}", name=model_name_ver)

    run["data_sample"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/results.csv")
    model_version["model"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/weights/best.pt")
    model_version.change_stage("production")

    results_assets = [
        f"{CURRENT_DATASET_DIR}/{model_name_ver}/labels.jpeg",
        f"{CURRENT_DATASET_DIR}/{model_name_ver}/labels_correlogram.jpeg",
        f"{CURRENT_DATASET_DIR}/{model_name_ver}/results.png",
    ]

    run["train/f1"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/F1_curve.png")

    run["val/conf_matrix"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/confusion_matrix.png")

    # You can also upload plot objects directly
    run["val/roc_curve"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/R_curve.png")
    run["val/p_curve"].upload(f"{CURRENT_DATASET_DIR}/{model_name_ver}/P_curve.png")

    data = pd.read_csv(f"{CURRENT_DATASET_DIR}/{model_name_ver}/results.csv", sep=",", skipinitialspace=True)
    for index, row in data.iterrows():
        epoch = index

        # Training metrics
        run["train/box_loss"].log(row["train/box_loss"], step=epoch)
        run["train/cls_loss"].log(row["train/cls_loss"], step=epoch)
        run["train/dfl_loss"].log(row["train/dfl_loss"], step=epoch)

        # Validation metrics
        run["val/box_loss"].log(row["val/box_loss"], step=epoch)
        run["val/cls_loss"].log(row["val/cls_loss"], step=epoch)
        run["val/dfl_loss"].log(row["val/dfl_loss"], step=epoch)

        # Learning rate metrics (log for each parameter group)
        run["lr/pg0"].log(row["lr/pg0"], step=epoch)
        run["lr/pg1"].log(row["lr/pg1"], step=epoch)
        run["lr/pg2"].log(row["lr/pg2"], step=epoch)

        # Evaluation metrics
        run["metrics/precision(B)"].log(row["metrics/precision(B)"], step=epoch)
        run["metrics/recall(B)"].log(row["metrics/recall(B)"], step=epoch)
        run["metrics/mAP50(B)"].log(row["metrics/mAP50(B)"], step=epoch)
        run["metrics/mAP50-95(B)"].log(row["metrics/mAP50-95(B)"], step=epoch)

        results_assets.append(f"{CURRENT_DATASET_DIR}/{model_name_ver}/train_batch{epoch}.jpg")

    run["results"].upload_files(results_assets)

    model_version.stop()
    run.stop()

    return {
        "status": "success",
        "exported_model": f"{CURRENT_DATASET_DIR}/{model_name_ver}/weights/best.pt"
    }
