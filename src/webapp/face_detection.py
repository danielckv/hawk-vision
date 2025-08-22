import supervision as sv
from torchvision.models import get_model

MODEL_ID = "people_counterv0/1"

model = get_model(MODEL_ID, api_key=API_KEY)

# Run inference
results = model.infer(image, model_id=MODEL_ID)
detections = sv.Detections.from_inference(results[0])

print(f"Found {len(detections)} people")

bbox_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT.colors[6],
    thickness=2
)

# Annotate our image with detections.
image_no_sahi = bbox_annotator.annotate(scene=image.copy(), detections=detections)

sv.plot_image(image_no_sahi)