
import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_mask(rgb_mask, colormap):
    output_mask = []

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask

if __name__ == "__main__":
    """ Create a directory """
    create_dir("mask")

    """ Dataset paths """
    dataset_path = "dataset"

    images = sorted(glob(os.path.join(dataset_path, "image", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "mask", "*.png")))

    print(f"Images: {len(images)}")
    print(f"RGB Masks: {len(masks)}")

    """ VOC 2012 dataset: colormap and class names """
    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor'
    ]

    """ Displaying the class name and its pixel value """
    for name, color in zip(VOC_CLASSES, VOC_COLORMAP):
        print(f"{name} - {color}")

    """ Loop over the images and masks """
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image and mask """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Resizing the image and mask """
        image = cv2.resize(image, (320, 320))
        mask = cv2.resize(mask, (320, 320))

        """ Processing the mask to one-hot mask """
        processed_mask = process_mask(mask, VOC_COLORMAP)

        """ Converting one-hot mask to single channel mask """
        grayscale_mask = np.argmax(processed_mask, axis=-1)
        grayscale_mask = (grayscale_mask / len(VOC_CLASSES)) * 255
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

        """ Saving the image """
        line = np.ones((320, 5, 3)) * 255
        cat_images = np.concatenate([
            image, line, mask, line,
            np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1)
        ], axis=1)

        cv2.imwrite(f"mask/{name}.png", cat_images)

 ##
