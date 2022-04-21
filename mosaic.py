import cv2
import numpy as np
import random
from math import sin 
import os 
import argparse
import secrets
from tqdm import trange

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_label(file):
    """
    file must be in yolo format 
    returns a list of labels from a text file
    [[class, xcenter, ycenter, width, height]]
    args: 
    """

    labels = []
    with open(file, 'r') as f:
        rows = f.readlines()
        for row in rows:
            class_id, x, y, w, h = row.split(' ')
            labels.append([int(class_id), float(x), float(y), float(w), float(h)])

    return labels 

def create_mosaic(image_files, label_files, mosaic_shape = (3, 3), output_shape = (640, 640)):
    """
    returns a mosaic of images with correct bounding boxes 
    !*!: output size is not exact

    args: 
        image_files: a list of image files to create the mosaic
        label_files: a list of labels corresponding to images
            structure: image_files = [image1.jpg, ...]
            label_files [image1.txt, ...] txt is in file
        mosaic_shape: shape to put mosaic into. len(images) must = product(mosaic_shape)
        output_size: output size of final mosaic (not exact) due to divis
    """
    assert len(image_files) == len(label_files), f'image_files and label_file are not of equal lenght'

    # make output size compateable with mosaic shape
    output_shape = tuple([output_shape[i] - output_shape[i] % mosaic_shape[i] for i in range(2)])

    # create shape of images 
    image_shape = tuple([output_shape[i] // mosaic_shape[i] for i in range(2)])

    # read and resize images 
    images = [cv2.imread(file) for file in image_files]
    labels = [parse_label(label) for label in label_files]

    # create empty image of mosaic 
    mosaic = np.zeros((*output_shape, 3), dtype = np.uint8)

    # create a new list of lables
    new_labels = []

    # loop through the images in random order
    order = np.random.permutation(len(images))

    for i in order:
        # calculte row and col
        row, col = i // mosaic_shape[0], i % mosaic_shape[0]

        # resize image 
        image = cv2.resize(images[i], image_shape, interpolation = cv2.INTER_AREA)

        # set slice of mosaic to image 
        row_start = row * image_shape[1] # w
        col_start = col * image_shape[0] # h
        mosaic[row_start:row_start + image_shape[0], col_start:col_start + image_shape[1]] = image[:, :]

        # create new labels 
        for label in labels[i]:
            
            class_id, x, y, w, h = label

            # scale and adjust bounding boxes 
            x = (col + x) / mosaic_shape[0]
            y = (row + y) / mosaic_shape[1]
            w /= mosaic_shape[0]
            h /= mosaic_shape[1]

            # add label to new labels 
            new_labels.append([class_id, x, y, w, h])

    return mosaic, new_labels

def draw_boxes(img, objects):
    """
    img: cv2/numpy in rgb order
    objects: numpy of shape (n, 6) where:
        0:4 are xyxy, 4 is confidence, 5 is class
    """
    # get height and width from the first 2 dims of img shape (h, w, c)
    height, width = img.shape[:2]
    # copy the image to prevent hazards 
    result = img.copy()

    

    for obj in objects:
        # create a color using sin and cos
        color = (100, 100, 100)

        xmin, ymin, xmax, ymax = int(obj[1] * width), int(obj[2] * height), int(obj[3] * width), int(obj[4] * height)

        # plot the rectangle and add a text label bellow
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(result, f"bird",
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5, color, 1, 2)

    return result

parser = argparse.ArgumentParser(description = 'expand dataset to include images')

parser.add_argument('--num-images', default = 20000, type = int,
    help = 'how many photos to expand the dataset by')

parser.add_argument('--image-folder', default = 'data/images/train',
    help = 'folder where images are stored')

parser.add_argument('--label-folder', default = 'data/labels/train',
    help = 'folder where labels are stored')


if __name__ == "__main__":
    args = parser.parse_args()

    num_images = args.num_images
    image_folder = args.image_folder
    label_folder = args.label_folder

    fileheads = [str(file).replace("<DirEntry '", '').replace(".jpg'>", '').replace(".JPG'>", '') for file in list(os.scandir(image_folder))]

    for image in trange(num_images):
        chosen_files = random.choices(fileheads, k = 4)
        image_files = [f'{image_folder}/{head}.jpg' for head in chosen_files]
        label_files = [f'{label_folder}/{head}.txt' for head in chosen_files]

        # new filename so large it would be ridicolous to have a collision
        new_filename = secrets.token_urlsafe(24) 

        try:
            mosaic, labels = create_mosaic(image_files, label_files, mosaic_shape=(2, 2))
        except:
            continue

        # save image
        cv2.imwrite(f'{image_folder}/{new_filename}.jpg', mosaic)

        # save labels
        with open(f'{label_folder}/{new_filename}.txt', 'w') as f:
            lines = [' '.join(map(str, label)) for label in labels]
            f.write('\n'.join(lines))



        





