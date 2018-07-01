import cv2
import numpy as np

from masterthesis.util.file import find_all_style_directories_path, find_all_file_paths, count_all_files

TRAIN_IMAGES_ROOT_PATH = '../../im/train/'
VALIDATION_IMAGES_ROOT_PATH = '../../im/validation/'
TEST_IMAGES_ROOT_PATH = '../../im/test/'
MAX_HEIGHT = 640
MAX_WIDTH = 480


def load_original_images(train):
    root_path = TRAIN_IMAGES_ROOT_PATH if train else VALIDATION_IMAGES_ROOT_PATH

    image_style_directories = find_all_style_directories_path(root_path)

    number_of_files = count_all_files(root_path, image_style_directories)

    image_index = 0
    x = np.empty((number_of_files, MAX_WIDTH, MAX_HEIGHT), dtype=np.float32)
    y = np.empty(number_of_files, dtype=np.int8)
    num_classes = 0

    for i, directory in enumerate(image_style_directories):
        file_paths = find_all_file_paths(root_path + directory)

        for fpath in file_paths:
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # plt.imshow(im_bw)
            # plt.show()

            # x[image_index, ...] = cv2.resize(im_bw, (MAX_WIDTH, MAX_HEIGHT))
            x[image_index, ...] = im_bw
            y[image_index] = i
            image_index += 1
        num_classes = num_classes + 1

        print('class: ' + str(i) + ' directory: ' + directory)

    return [x, y, num_classes]


def load_test_data():
    image_style_directories = find_all_style_directories_path(TEST_IMAGES_ROOT_PATH)

    number_of_files = count_all_files(TEST_IMAGES_ROOT_PATH, image_style_directories)

    image_index = 0
    x = np.empty((number_of_files, MAX_WIDTH, MAX_HEIGHT), dtype=np.float32)
    y = np.empty(number_of_files, dtype=np.int8)
    num_classes = 0

    for i, directory in enumerate(image_style_directories):
        file_paths = find_all_file_paths(TEST_IMAGES_ROOT_PATH + directory)

        for fpath in file_paths:
            image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            x[image_index, ...] = im_bw
            y[image_index] = i
            image_index += 1
        num_classes = num_classes + 1

        print('class: ' + str(i) + ' directory: ' + directory)

    return [x, y, num_classes]
