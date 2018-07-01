import glob
import os


def find_all_file_paths(path):
    file_paths = []

    file_types = ('.jpg', '.JPG', '.png')

    for file_type in file_types:
        for file in glob.glob(path + "**/*" + file_type, recursive=True):
            file_paths.append(file)

    return file_paths


def find_all_style_directories_path(root_path):
    for root, dirs, files in os.walk(root_path):
        return dirs


def count_all_files(images_root_path, directories):
    number_of_files = 0
    for directory in directories:
        number_of_files += number_of_files_in_directory(images_root_path + directory)
    return number_of_files


def number_of_files_in_directory(path):
    return len(find_all_file_paths(path))

