import os
import shutil
import uuid

from PIL import ImageDraw, Image
from samgeo import split_raster
from samgeo.text_sam import LangSAM

from algorithms.io.geotiff import GeoTiffHandler


class SegmentAnything:
    def __init__(self, model=None):
        if model is not None:
            self.model = LangSAM(model_type=model)
        else:
            self.model = LangSAM()

        # get parent directory
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        self.current_session_id = uuid.uuid4()
        self.current_session_path = f"{parent_dir}/../results/{self.current_session_id}"

        os.makedirs(self.current_session_path, exist_ok=True)
        print(f"working on device: {self.model.device}")

    def predict_image(self, nlp_phrase, image_file: str):
        results = self.model.predict(
            image=image_file,
            text_prompt=nlp_phrase,
            box_threshold=0.22,
            text_threshold=0.30,
            mask_multiplier=255,
            return_results=True,
            verbose=True,
        )
        self.model.show_anns(output=f"{self.current_session_path}/merged.png")
        results = [(self.model.masks, self.model.boxes, self.model.phrases, self.model.logits)]

        return results

    def predict_tiff(self, nlp_phrase, geotiff: GeoTiffHandler):
        os.makedirs(f"{self.current_session_path}/tiles", exist_ok=True)
        print(f"Splitting raster {geotiff.path}")
        split_raster(geotiff.path, tile_size=512, out_dir=f"{self.current_session_path}/tiles")
        print(f"Splitting done for {geotiff.path}")
        print(f"Processing file: {geotiff.path}")
        self.model.predict_batch(
            images=f"{self.current_session_path}/tiles",
            out_dir=f"{self.current_session_path}/masks",
            text_prompt=nlp_phrase,
            box_threshold=0.30,
            text_threshold=0.30,
            mask_multiplier=255,
            verbose=True,
        )
        print(f"Processing done for {geotiff.path}")

        print("Saving boxes...")
        self.model.save_boxes(f"{self.current_session_path}/merged_boxes.vectors")
        self.model.raster_to_vector(f"{self.current_session_path}/masks/merged.tif",
                                    f"{self.current_session_path}/merged.geojson")

        self.generate_annotation_image_from_origin(geotiff.path, f"{self.current_session_path}/merged.png")

        results = [(self.model.masks, self.model.boxes, self.model.phrases, self.model.logits)]

        print("Full cycle finished.")
        shutil.move(f"{self.current_session_path}/masks/merged.tif", f"{self.current_session_path}/merged.tiff")
        shutil.rmtree(f"{self.current_session_path}/tiles", ignore_errors=True)
        shutil.rmtree(f"{self.current_session_path}/masks", ignore_errors=True)
        return results

    def generate_annotation_image_from_origin(self, origin_image_path, annotation_image_path):
        origin_pillow = Image.open(origin_image_path)
        for box in self.model.boxes:
            box = box.cpu().numpy()
            draw = ImageDraw.Draw(origin_pillow)
            draw.rectangle(box, outline='red')
        origin_pillow.save(annotation_image_path)

    def get_solution_results(self):
        return f"{self.current_session_path}/"
