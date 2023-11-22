import csv
import json
import os
from PIL import Image

# Set the input CSV file path, image directory path, and output directory path
input_csv_file = '/home/omar/Documents/annotated_images.csv'
image_dir = '/home/omar/Documents/ximg'
output_dir = '/home/omar/Documents/xres'

# Read data from the CSV file
gt_data_dict = {}
with open(input_csv_file, 'r') as csvfile:
    # Create a CSV reader
    reader = csv.reader(csvfile, delimiter=',')
    # Skip the header row
    next(reader)
    for row in reader:
        filename = row[1]
        bbox = [round(float(x)) for x in row[2][1:-1].split(',')]  # json.loads(row[2])
        img_path = os.path.join(image_dir, filename)
        if img_path not in gt_data_dict:
            gt_data_dict[img_path] = {
                'text_line': []
            }
        gt_data_dict[img_path]['text_line'].append({
            'confidence': 1,
            'polygon': [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        })

# Get image size for each image
for img_path in gt_data_dict:
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        gt_data_dict[img_path]['img_size'] = [img_height, img_width]

# Write ground truth data to JSON files
for img_path, gt_data in gt_data_dict.items():
    output_json_file = os.path.splitext(os.path.basename(img_path))[0] + '.json'
    output_json_path = os.path.join(output_dir, output_json_file)
    with open(output_json_path, 'w') as jsonfile:
        json.dump(gt_data, jsonfile, indent=4)
