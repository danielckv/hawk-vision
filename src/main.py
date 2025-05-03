from langrs import LangRS


def main():
    # The class accepts tif/ RGB images
    text_input = "find all rooftops in this area."

    # Path to the input remote sensing image
    image_input = "./image_15cm.tif"

    # Initialize LangRS with the input image, text prompt, and output directory
    langrs = LangRS(image_input, text_input, "output_folder")

    # Detect bounding boxes using the sliding window approach with example parameters
    bounding_boxes = langrs.generate_boxes(window_size=600, overlap=300, box_threshold=0.25, text_threshold=0.25)

    bboxes_filtered = langrs.outlier_rejection()

    # Retreive certain bounding boxes
    bboxes_zscore = bboxes_filtered['zscore']

    # Generate segmentation masks for the filtered bounding boxes of the provided key
    masks = langrs.generate_masks(boxes=bounding_boxes)
    # Or
    masks = langrs.generate_masks(boxes=bboxes_zscore)


if __name__ == "__main__":
    main()
