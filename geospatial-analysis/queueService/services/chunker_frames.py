import argparse
import os
from itertools import product

import cv2
import numpy as np
from local_queue_processor import enqueue_video_tiles
from PIL import Image
from scipy.ndimage import zoom


def chunk_video_file_frames(source, dest_folder):
    cap = cv2.VideoCapture(source)
    path_to_save = dest_folder

    current_frame = 1

    if cap.isOpened() == False:
        print('Cap is not open')
        return

    # cap opened successfully
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            name = 'frame_' + str(current_frame) + '.jpg'
            print(f'Creating: {name}')
            cv2.imwrite(os.path.join(path_to_save, name), frame)
            tile(os.path.join(path_to_save, name), 150)
            # enqueue_video_tiles(path_to_save)
            current_frame += 1
        else:
            break

    # release capture 
    cap.release()
    print('done')


def tile(file_name, d):
    name, ext = os.path.splitext(file_name)
    img = Image.open(file_name)
    w, h = img.size

    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = f'{name}_tile_{i}_{j}{ext}'
        cropped_image = img.crop(box)
        zoomed_image = cv2_clipped_zoom(np.array(cropped_image), 0)
        cv2.imwrite(out, zoomed_image)


def cv2_clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def video_tiles_processor(tiles_path):
    print(f'Processing tiles: {tiles_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunk video file into frames')
    parser.add_argument('source', help='path to video file')
    parser.add_argument('dest_folder', help='path to folder where frames will be saved')
    args = parser.parse_args()
    chunk_video_file_frames(args.source, args.dest_folder)
