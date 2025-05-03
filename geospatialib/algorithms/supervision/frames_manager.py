import cv2
import numpy as np


def split_and_zoom_frame(frame, rows, cols, zoom_factor=1.5):
    # Get the dimensions of the image
    height, width, _ = frame.shape

    # Calculate the size of each section
    section_height = height // rows
    section_width = width // cols

    # List to hold the zoomed sections
    zoomed_sections = []

    # Split and zoom the frame
    for row in range(rows):
        for col in range(cols):
            x_start = col * section_width
            y_start = row * section_height
            x_end = x_start + section_width
            y_end = y_start + section_height
            section = frame[y_start:y_end, x_start:x_end]

            # Zooming the section
            zoomed_section = cv2.resize(section, None, fx=zoom_factor, fy=zoom_factor)
            zoomed_sections.append(zoomed_section)

    return zoomed_sections


def reassemble_frame(sections, original_shape, rows, cols):
    # Create a blank frame with the original dimensions
    reassembled_frame = np.zeros(original_shape, dtype=np.uint8)

    # Calculate the size of each original section
    section_height = original_shape[0] // rows
    section_width = original_shape[1] // cols

    # Iterate over the sections and place them in the correct position
    for i, section in enumerate(sections):
        row = i // cols
        col = i % cols
        x_start = col * section_width
        y_start = row * section_height

        # Resize the section back to its original size if it was zoomed
        resized_section = cv2.resize(section, (section_width, section_height))

        # Place the section
        reassembled_frame[y_start:y_start + section_height, x_start:x_start + section_width] = resized_section

    return reassembled_frame