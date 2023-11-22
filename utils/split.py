import os, shutil
import random

train_folder = "/home/omar/Documents/new elya data/dataset3/split_png/train"
val_folder = "/home/omar/Documents/new elya data/dataset3/split_png/val"
test_folder = "/home/omar/Documents/new elya data/dataset3/split_png/test"
json_folder = "/home/omar/Documents/new elya data/dataset3/labels_json"
new_folder = "/home/omar/Documents/new elya data/dataset3/images_png"


def split_train_val_test_folders_one_line(new_folder, json_folder, train_folder, val_folder, test_folder):
    """
    Split the dataset to train, val and test sets
    :param new_folder: the path of the images folder
    :param json_folder: the path to ground-truth json folder
    :param train_folder: the path to train folder
    :param val_folder: the path to val folder
    :param test_folder: the path to test folder
    :return:
    """
    # Create the train, val, and test folders if they don't exist
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get the list of files in the new folder
    train_list = []
    val_list = []
    test_list = []
    all_files = os.listdir(new_folder)
    for font in ["IBMPlexSansArabic-Light", "IBMPlexSansArabic-Medium", "M_Unicode_Hadeel_Regular",
                 "NotoNaskhArabic-Bold",
                 "DINNextLTArabic-Regular-3", "Bahij_Myriad_Arabic-Regular", "badiya-lt-regular",
                 "Al-QuranAlKareem_Regular",
                 "18_Khebrat_Musamim_Bold", "Amiri-Italic"]:
        for noise in ["Quasicrystal", "Plain_white", "Gaussian_Noise"]:
            files = [file for file in all_files if font + "_" + noise in file]
            # Shuffle the files
            random.shuffle(files)
            # Split the shuffled files into train, validation, and test lists
            train_list.extend(files[:3])
            val_list.extend(files[3:5])
            test_list.extend(files[5:6])

        images = "images"
        json_labels = "labels_json"
        # Loop through the train files
        for image_file in train_list:
            image_path = os.path.join(new_folder, image_file)
            image_base, image_ext = os.path.splitext(image_file)
            json_file = image_base + ".json"
            # json_file = image_file.replace(".jpg", ".json")  # Assuming image and json files have the same name
            json_path = os.path.join(json_folder, json_file)
            if os.path.isfile(json_path):
                # If a corresponding JSON file exists in the labels JSON folder, copy the image and JSON file to the train folder
                new_image_path = os.path.join(train_folder, images, image_file)
                shutil.copy2(image_path, new_image_path)
                new_json_path = os.path.join(train_folder, json_labels, json_file)
                shutil.copy2(json_path, new_json_path)

        # Loop through the val files
        for image_file in val_list:
            image_path = os.path.join(new_folder, image_file)
            image_base, image_ext = os.path.splitext(image_file)
            json_file = image_base + ".json"
            # json_file = image_file.replace(".jpg", ".json")  # Assuming image and json files have the same name
            json_path = os.path.join(json_folder, json_file)
            if os.path.isfile(json_path):
                # If a corresponding JSON file exists in the labels JSON folder, copy the image and JSON file to the val folder
                new_image_path = os.path.join(val_folder, images, image_file)
                shutil.copy2(image_path, new_image_path)
                new_json_path = os.path.join(val_folder, json_labels, json_file)
                shutil.copy2(json_path, new_json_path)

        # Loop through the test files
        for image_file in test_list:
            image_path = os.path.join(new_folder, image_file)
            image_base, image_ext = os.path.splitext(image_file)
            json_file = image_base + ".json"
            # json_file = image_file.replace(".jpg", ".json")  # Assuming image and json files have the same name
            json_path = os.path.join(json_folder, json_file)
            if os.path.isfile(json_path):
                # If a corresponding JSON file exists in the labels JSON folder, copy the image and JSON file to the test folder
                new_image_path = os.path.join(test_folder, images, image_file)
                shutil.copy2(image_path, new_image_path)
                new_json_path = os.path.join(test_folder, json_labels, json_file)
                shutil.copy2(json_path, new_json_path)


if __name__ == '__main__':
    # Load image
    split_train_val_test_folders_one_line(new_folder, json_folder, train_folder, val_folder, test_folder)
