import os
import csv
import shutil
import random
import json
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from pdf2image import convert_from_path
import cv2
import numpy as np
from tqdm import tqdm

from doc_ufcn_0_1_8.doc_ufcn.train import mask
from doc_ufcn_0_1_8.doc_ufcn import image
from doc_ufcn_0_1_8.doc_ufcn import models

_, parameters = models.download_model('generic-historical-line')

IMAGE_PATH = "/home/omar/Desktop/elya_data/datasets/page/images/page-09.png"  # /home/omar/Music/ff/13278_1.png"#"I6SLT-9
GT_PATH = "/home/omar/Desktop/elya_data/datasets/page/labels_json"
PRED_PATH = "/home/omar/Desktop/Doc_ufcn/PRODS/fine_tuning_docufcn_on_new_reviewed_dataset/prediction/test/page"
FILE_NAME = "page-09.json"


def to_csv(image_folder, label_folder, output_csv):
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image File', 'Label File'])

        for i, image_file in enumerate(image_files):
            label_file = label_files[i]
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, label_file)
            writer.writerow([image_path, label_path])


def to_png(input_folder_path, output_folder_path):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if
                   f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith(
                       '.tif') or f.endswith('.tiff') or f.endswith('.JPG') or f.endswith('.jp2')]
    for image_file in tqdm(image_files):
        # Construct the input and output file paths
        input_file_path = os.path.join(input_folder_path, image_file)
        output_file_path = os.path.join(output_folder_path, os.path.splitext(image_file)[0] + '.png')

        # Load the input image
        input_image = Image.open(input_file_path)

        # Convert the image to PNG format and save it
        input_image.save(output_file_path, "PNG")


def to_tiff(input_image_path="/home/omar/20776_1.jpg", output_image_path="/home/omar/20776_1.tif"):
    # Load the input image
    input_image = Image.open(input_image_path)
    # Convert the image to TIFF format and save it
    input_image.save(output_image_path, "TIFF")  # "PNG"


def read_gts(imgs_path, gt_path, pred_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for filename in os.listdir(imgs_path):
        # Load image
        if filename.endswith('.png'):
            image_ = cv2.cvtColor(cv2.imread(os.path.join(imgs_path, filename)), cv2.COLOR_BGR2RGB)
            read_gt(save_path, image_, gt_path, pred_path, filename.replace('.png', '.json'), plot=True,
                    with_brect=False)


def read_gt(save_path, img, gt_path, pred_path, file_name, plot=True, with_brect=False, thickness=4):
    """
    Read ground truth (gt) and predicted labels from JSON files and overlay them on the input image.
    :param img: Input image in BGR format.
    :param gt_path: Path to the directory containing the ground truth (gt) JSON files.
    :param pred_path: Path to the directory containing the predicted JSON files.
    :param file_name: Name of the JSON file for the current image.
    :param plot: Whether to display the overlaid image with matplotlib.
    :param with_brect: Whether to draw bounding rectangles around the regions.
    :param thickness: Thickness of the lines used for drawing contours.
    :return: Two tuples defines each a list of tuples representing the coordinates of the bounding rectangles for both gt and predicted regions.
        Each tuple contains (x, y, width, height).
    """
    gt_rect_coords = []
    pred_rect_coords = []
    fig, ax = plt.subplots()
    with open(os.path.join(gt_path, file_name), "r") as file:
        with open(os.path.join(pred_path, file_name), "r") as f:
            gt_regions = json.load(file)
            # main.predict(img)-->overlap differs from below plotted gt polygon
            for dic in gt_regions['text_line']:
                gt_points = dic['polygon']
                #cv2.drawContours(img, [np.array(gt_points)], 0, [255, 0, 0], 2)
            pred_regions = json.load(f)
            for dic in pred_regions['text_line']:
                pred_points = dic['polygon']
                #cv2.drawContours(img, [np.array(pred_points)], 0, [0, 255, 0], 1)"""
            ax.imshow(img, cmap='gray')
            if with_brect:
                """for dic in gt_regions['text_line']:
                    gt_points = dic['polygon']
                    x, y, w, h = cv2.boundingRect(np.array(gt_points))
                    gt_rect_coords.append((x, y, w, h))
                    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)"""
                for dic in pred_regions['text_line']:
                    pred_points = dic['polygon']
                    x, y, w, h = cv2.boundingRect(np.array(pred_points))
                    pred_rect_coords.append((x, y, w, h))
                    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
            if plot:
                plt.axis('off')
                plt.show()
                plt.rcParams["figure.figsize"] = (20, 15)
                # Remove axes

                # plt.savefig(os.path.join(save_path, file_name.replace('.json', '.png')), bbox_inches='tight')
                plt.close(fig)
            return gt_rect_coords, pred_rect_coords


def gen_masks(images_dir, input_dir, output_dir, network_size=None):
    """
    Generate label masks from their corresponding json ground-truth files
    :param images_dir: the path of the images folder
    :param input_dir: the path of the json ground-truth folder of the corresponding images
    :param output_dir: the path of the mask folder
    :param network_size: Default None or specify where we want to get resized label mask
    :return:
    """
    # Create the labels folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.json'):
            # Load the XML file
            json_path = os.path.join(input_dir, filename)
            with open(json_path, "r") as file:
                list = []
                gt_regions = json.load(file)
                for dic in gt_regions['text_line']:
                    gt_points = dic['polygon']
                    list.append(gt_points)
            label_polygons = {"text_line": list}
            for filename_ in os.listdir(images_dir):
                if os.path.splitext(filename)[0] == os.path.splitext(filename_)[0]:
                    ext = os.path.splitext(filename_)[1]
                    break
            output_filename = os.path.splitext(filename)[0] + ext
            output_path = os.path.join(output_dir, output_filename)
            mask.generate_mask(
                gt_regions['img_size'][1], gt_regions['img_size'][0], network_size, label_polygons,
                {"text_line": (0, 0, 255)}, output_path
            )


def ann_img(images_folder, json_folder, new_folder):
    """
    Get only the annotated images with their corresponding labels_json files
    :param images_folder:
    :param json_folder:
    :param new_folder:
    :return:
    """
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Loop through the images folder
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        image_base, image_ext = os.path.splitext(image_file)
        json_file = image_base + ".json"  # Assuming image and json files have the same base name
        json_path = os.path.join(json_folder, json_file)
        if os.path.isfile(json_path):
            # If a corresponding JSON file exists in the labels JSON folder, copy the image to the new folder
            new_image_path = os.path.join(new_folder, image_file)
            shutil.copy2(image_path, new_image_path)


def copy(images_folder, json_folder, train_folder):
    """

    :return:
    """
    all_files = os.listdir(images_folder)
    for image_file in all_files:
        image_path = os.path.join(images_folder, image_file)
        image_base, image_ext = os.path.splitext(image_file)
        json_file = image_base + ".json"
        # json_file = image_file.replace(".jpg", ".json")  # Assuming image and json files have the same name
        json_path = os.path.join(json_folder, json_file)
        if os.path.isfile(json_path):
            # If a corresponding JSON file exists in the labels JSON folder, copy the image and JSON file to the train folder
            new_json_path = os.path.join(train_folder, json_file)
            shutil.copy2(json_path, new_json_path)


def split_train_val_test_folders(new_folder, json_folder, train_folder, val_folder, test_folder):
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
    all_files = os.listdir(new_folder)

    # Calculate the number of files for train, val, and test
    num_files = len(all_files)
    num_train = int(num_files * 0.8)
    num_val = int(num_files * 0.1)
    num_test = num_files - num_train - num_val

    # Randomly shuffle the files
    random.shuffle(all_files)

    # Split the files into train, val, and test
    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]
    images = "images"
    json_labels = "labels_json"
    # Loop through the train files
    for image_file in train_files:
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
            # label
            new_label_path = os.path.join(train_folder, "labels", image_file)
            shutil.copy2(os.path.join("/home/omar/Music/collected_data/arabictype.wordpress.com/labels", image_file),
                         new_label_path)

    # Loop through the val files
    for image_file in val_files:
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
            # label
            new_label_path = os.path.join(val_folder, "labels", image_file)
            shutil.copy2(os.path.join("/home/omar/Music/collected_data/arabictype.wordpress.com/labels", image_file),
                         new_label_path)

    # Loop through the test files
    for image_file in test_files:
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


def resizing_image(network_size, images_folder, output_folder):
    """
    resize input images to the correspond network size
    :param network_size: the size of the network
    :param images_folder: the path of input images folder
    :param output_folder: the path of the resized images folder
    :return:
    """
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        input_image = cv2.imread(image_path)
        assert isinstance(
            input_image, np.ndarray
        ), "Input image must be an np.array in RGB"
        input_image = np.asarray(input_image)
        if len(input_image.shape) < 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

        output_path = os.path.join(output_folder, image_file)
        # Resize the image
        resized_image, padding = image.resize(input_image, network_size, parameters["mean"])
        cv2.imwrite(output_path, resized_image)


def averaging_results(folder_path, csv_file_path):
    # Initialize variables to store cumulative values
    sum_iou = 0.0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_fscore = 0.0
    sum_miou = 0.0
    sum_ap_5 = 0.0
    sum_ap_75 = 0.0
    sum_ap_95 = 0.0
    sum_ap_5_95 = 0.0
    count = 0

    # Iterate over the JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Open and load the JSON file
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            # Extract the values from the JSON
            text_line = data["text_line"]
            iou = text_line["iou"]
            precision = text_line["precision"]
            recall = text_line["recall"]
            fscore = text_line["fscore"]
            miou = text_line["miou"]
            ap_5 = text_line["AP@[.5]"]
            ap_75 = text_line["AP@[.75]"]
            ap_95 = text_line["AP@[.95]"]
            ap_5_95 = text_line["AP@[.5,.95]"]

            # Update cumulative values
            sum_iou += iou
            sum_precision += precision
            sum_recall += recall
            sum_fscore += fscore
            sum_miou += miou
            sum_ap_5 += ap_5
            sum_ap_75 += ap_75
            sum_ap_95 += ap_95
            sum_ap_5_95 += ap_5_95
            count += 1

    # Calculate the average of the overall results
    avg_iou = sum_iou / count
    avg_precision = sum_precision / count
    avg_recall = sum_recall / count
    avg_fscore = sum_fscore / count
    avg_miou = sum_miou / count
    avg_ap_5 = sum_ap_5 / count
    avg_ap_75 = sum_ap_75 / count
    avg_ap_95 = sum_ap_95 / count
    avg_ap_5_95 = sum_ap_5_95 / count

    # Create a list of dictionaries for CSV
    results = [
        {
            "Metric": "iou",
            "Value": avg_iou
        },
        {
            "Metric": "precision",
            "Value": avg_precision
        },
        {
            "Metric": "recall",
            "Value": avg_recall
        },
        {
            "Metric": "fscore",
            "Value": avg_fscore
        },
        {
            "Metric": "miou",
            "Value": avg_miou
        },
        {
            "Metric": "AP@[.5]",
            "Value": avg_ap_5
        },
        {
            "Metric": "AP@[.75]",
            "Value": avg_ap_75
        },
        {
            "Metric": "AP@[.95]",
            "Value": avg_ap_95
        },
        {
            "Metric": "AP@[.5,.95]",
            "Value": avg_ap_5_95
        }
    ]

    # Save results to CSV file
    fieldnames = ["Metric", "Value"]
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def pdf_to_images(pdf_path, output_dir):
    # Convert PDF pages to PIL Image objects
    images = convert_from_path(pdf_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each image as a PNG file
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"image_" + pdf_path.split("/")[-1][:-4] + "_" + str(i + 1) + ".png")
        image.save(image_path, "PNG")
        print(f"Saved image: {image_path}")


def sum_lines(folder_path, name):
    # Iterate over the JSON files in the folder
    sum = 0
    num_images = len(os.listdir(folder_path))
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Open and load the JSON file
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                sum += len(data['text_line'])
    print(f"Number of lines in this data folder= {sum} lines in {num_images} images for {name}")
    return sum


def random_crop(image_path, label_path, output_path):
    random_seed = 42
    random.seed(random_seed)
    # Define the transformation pipeline with random crop
    transform = A.Compose([
        A.CenterCrop(width=650, height=256, ),  # Set the desired crop size and seed
    ])
    # crop_params = transform.get_params()
    image = Image.open(image_path)
    transformed_image = transform(image=np.array(image))["image"]
    transformed_image = Image.fromarray(transformed_image)
    filename = os.path.basename(image_path).split(".")[0] + ".png"
    output_image_path = os.path.join(output_path, "images", filename)
    transformed_image.save(output_image_path)
    # save label
    label = Image.open(label_path)
    transformed_image1 = transform(image=np.array(label))["image"]
    transformed_image1 = Image.fromarray(transformed_image1)
    filename = os.path.basename(label_path).split(".")[0] + ".png"
    output_label_path = os.path.join(output_path, "labels", filename)
    transformed_image1.save(output_label_path)
    """plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.title("Transformed Image")
    plt.show()"""


def pixel_dropout(image_path, label_path, output_path):
    random_seed = 42
    random.seed(random_seed)
    # Define the transformation pipeline with random crop
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    # crop_params = transform.get_params()
    image = Image.open(image_path)
    transformed_image = transform(image=np.array(image))["image"]
    transformed_image = Image.fromarray(transformed_image)
    filename = os.path.basename(image_path).split(".")[0] + ".png"
    output_image_path = os.path.join(output_path, "images", filename)
    transformed_image.save(output_image_path)
    # save label
    label = Image.open(label_path)
    transformed_image1 = transform(image=np.array(label))["image"]
    transformed_image1 = Image.fromarray(transformed_image1)
    filename = os.path.basename(label_path).split(".")[0] + ".png"
    output_label_path = os.path.join(output_path, "labels", filename)
    transformed_image1.save(output_label_path)


def ensemble_crop(images_path, labels_path, output_path):
    for filename in os.listdir(images_path):
        if filename.endswith(".png"):
            file_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, filename)
            # random_crop(file_path, label_path, output_path)
            pixel_dropout(file_path, label_path, output_path)


def resize(images_path, denoised_path, output_path):
    for filename in os.listdir(images_path):
        file_path = os.path.join(images_path, filename)
        img = Image.open(file_path)
        size = img.size
        image = Image.open(os.path.join(denoised_path, filename))
        resized = image.resize(size)
        out_path = os.path.join(output_path, filename)
        resized.save(out_path)


if __name__ == '__main__':
    """to_csv("/home/omar/Documents/elyadata_ds_p_n_g/val/images",
           "/home/omar/Documents/elyadata_ds_p_n_g/val/labels",
           '/home/omar/Documents/elyadata_ds_p_n_g/val/val.csv')"""
    # Load image
    image_ = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    #read_gt("/home/omar/Videos", image_, GT_PATH, PRED_PATH, FILE_NAME, with_brect=True)
    # to_png("/home/omar/Music/val_new", "/home/omar/Music/val_new")
    # to_png("/home/omar/Documents/out", "/home/omar/Documents/out_png")
    path = "/home/omar/Documents/jj6"
    gt_path = "/home/omar/Documents/Quran-65-pages_json"
    # read_gts(path, path, path, "/home/omar/Documents/jj6/labeled_images")
    """read_gts("/home/omar/Documents/yarambouk_images_p_n_g",
             "/home/omar/Documents/logs/predictions/test/thres0/preds",
             "/home/omar/Documents/logs/predictions/test/thres0/preds",
             "/home/omar/Documents/logs/predictions/test/thres0/pred_images")
    copy("/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/val/images",
          "/home/omar/Desktop/nn3",
         "/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/val/labels_json")
    copy("/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/train/images",
         "/home/omar/Desktop/nn3",
         "/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/train/labels_json")
    copy("/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/test/images",
         "/home/omar/Desktop/nn3",
         "/home/omar/Desktop/elya_data/datasets/__zzlip/yarmouk/test/labels_json")
    #read_gt("/home/omar/Videos", image_, GT_PATH, PRED_PATH, FILE_NAME)
    gen_masks("/home/omar/Documents/quran_split/train/images",
              "/home/omar/Documents/quran_split/train/labels_json",
              "/home/omar/Documents/quran_split/train/labels")
    gen_masks("/home/omar/Music/val_new",
              "/home/omar/Music/val_json",
              "/home/omar/Music/labels")"""
    """gen_masks("/home/omar/Documents/new elya data/dataset3/split_png/train/images",
              "/home/omar/Documents/new elya data/dataset3/split_png/train/labels_json",
              "/home/omar/Documents/new elya data/dataset3/split_png/train/labels")"""

    # Specify the paths of the folders with images and JSON files
    images_folder = "/home/omar/Desktop/elya_data/images"
    json_folder = "/home/omar/Desktop/elya_data/labels_json"
    # Specify the path of the new folder to create
    new_folder = "/home/omar/Desktop/elya_data/annotated_images"
    # ann_img(images_folder, json_folder, new_folder)

    # Specify the paths of the train, val, and test folders to create
    train_folder = "/home/omar/RASM/train"
    val_folder = "/home/omar/RASM/val"
    test_folder = "/home/omar/RASM/test"
    # split_train_val_test_folders("/home/omar/Documents/quran", "/home/omar/Documents/quran_json", "/home/omar/Documents/quran_split/train", "/home/omar/Documents/quran_split/val", "/home/omar/Documents/quran_split/test")
    # split_train_val_test_folders("/home/omar/Desktop/elya_data/datasets/ALQistas/images", "/home/omar/Desktop/elya_data/datasets/ALQistas/labels_json", "/home/omar/Desktop/elya_data/datasets/__zzplit/ALQistas/train", "/home/omar/Desktop/elya_data/datasets/__zzplit/ALQistas/val", "/home/omar/Desktop/elya_data/datasets/__zzplit/ALQistas/test")
    # split_train_val_test_folders("/home/omar/Desktop/elya_data/datasets/I6/images", "/home/omar/Desktop/elya_data/datasets/I6/pre-annotated-labels_json", "/home/omar/Desktop/elya_data/datasets/__zzplit/I6/train", "/home/omar/Desktop/elya_data/datasets/__zzplit/I6/val", "/home/omar/Desktop/elya_data/datasets/__zzplit/I6/test")
    # split_train_val_test_folders("/home/omar/Desktop/elya_data/datasets/page/images", "/home/omar/Desktop/elya_data/datasets/page/labels_json", "/home/omar/Desktop/elya_data/datasets/__zzplit/page/train", "/home/omar/Desktop/elya_data/datasets/__zzplit/page/val", "/home/omar/Desktop/elya_data/datasets/__zzplit/page/test")
    # split_train_val_test_folders("/home/omar/Desktop/elya_data/datasets/picture/images", "/home/omar/Desktop/elya_data/datasets/picture/labels_json", "/home/omar/Desktop/elya_data/datasets/__zzplit/picture/train", "/home/omar/Desktop/elya_data/datasets/__zzplit/picture/val", "/home/omar/Desktop/elya_data/datasets/__zzplit/picture/test")
    # split_train_val_test_folders("/home/omar/Music/collected_data/arabictype.wordpress.com/images", "/home/omar/Music/collected_data/arabictype.wordpress.com/labels_json", "/home/omar/Desktop/elya_data/datasets/__zzlip1/almada/train", "/home/omar/Desktop/elya_data/datasets/__zzlip1/almada/val", "/home/omar/Desktop/elya_data/datasets/__zzlip1/almada/test")

    # resizing_image(768, "/home/omar/Documents/elyadata_ds/train/images", "/home/omar/Music/elyadata_ds_res2/train/images")
    # resizing_image(768, "/home/omar/Documents/elyadata_ds/val/images", "/home/omar/Music/elyadata_ds_res2/val/images")
    # to_tiff()

    # Example usage
    # pdf_to_images("/home/omar/Videos/generate_syn/pymupdf/1.pdf", "/home/omar/Pictures")
    """for i in ["1822_11"]:
        pdf_file = "/home/omar/Music/collected_data/almada/"+str(i)+".pdf"
        output_directory = "/home/omar/Music/collected_data/almada/images"
        pdf_to_images(pdf_file, output_directory)"""

    # Folder path containing the JSON files
    folder_path = "/home/omar/Desktop/Doc_ufcn/PRODS/fine_tuning_docufcn_on_new_reviewed_dataset/evaluation/test"
    csv_file_path = f"{folder_path}/avg_results_14june.csv"
    #averaging_results(folder_path, csv_file_path)
    for dataset in ["aco", "almada", "ALQistas", "I6", "I6_complement", "page", "picture", "quran", "yarmouk",
                    "one_ayah", "one_line"]:
        root_path = f"/home/omar/Desktop/elya_data/datasets/{dataset}"
        if True:#dataset in ["almada", "one_ayah", "one_line"]:
            sum_lines(f"/home/omar/Desktop/elya_data/datasets/__zzlip1/{dataset}/train/labels_json", dataset)
            sum_lines(f"/home/omar/Desktop/elya_data/datasets/__zzlip1/{dataset}/val/labels_json", dataset)
            sum_lines(f"/home/omar/Desktop/elya_data/datasets/__zzlip1/{dataset}/test/labels_json", dataset)
            continue
        if os.path.exists(f"{root_path}/labels_json"):
            sum_lines(f"{root_path}/labels_json", dataset)
        else:
            sum_lines(f"{root_path}/pre-annotated-labels_json", dataset)
    #sum_lines("/home/omar/Desktop/Doc_ufcn/PRODS","191")
    """random_crop("/home/omar/Desktop/elya_data/datasets/__zzlip1/yarmouk/val/images/4155_1.png",
                "/home/omar/Desktop/elya_data/datasets/__zzlip1/yarmouk/val/labels/4155_1.png",
                "/home/omar/Videos/generate_syn/cropped")"""
    """ensemble_crop("/home/omar/Desktop/elya_data/datasets/__zzlip1/yarmouk/val/images",
                  "/home/omar/Desktop/elya_data/datasets/__zzlip1/yarmouk/val/labels",
                  "/home/omar/Videos/generate_syn/cropped/flip")"""
    """resize("/home/omar/Videos/generate_syn/investigate/testing_impact/test/images",
           "/home/omar/Videos/generate_syn/investigate/denoised 9alo/test/images",
           "/home/omar/Videos/generate_syn/investigate/denoised 9alo/test/resized")"""
