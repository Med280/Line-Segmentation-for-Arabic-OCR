import csv
import os
from PIL import Image
import json

# Dictionary to store image data
image_data = {}
image_folder = "/home/omar/Music/val_new"
output_dir = "/home/omar/Music/val_json"
# Read the CSV file
with open('/home/omar/Downloads/del.csv', 'r') as file:
    csv_reader = csv.reader(file)

    # Skip the header row
    next(csv_reader)

    # Iterate over each row in the CSV
    for row in csv_reader:
        filename = row[0]
        data_dict = json.loads(row[5])
        x = data_dict['x']
        y = data_dict['y']
        width = data_dict['width']
        height = data_dict['height']

        # If the image is not present in the dictionary, create an empty list
        if filename not in image_data:
            image_data[filename] = []

        # Append the coordinates to the image's list
        image_data[filename].append((x, y, width, height))

# Print the image data
for filename, coordinates in image_data.items():
    output_dict = {
        'text_line': [],
        'img_size': []
    }
    for line_cord in coordinates:
        output_dict['text_line'].append({
            'confidence': 1,
            'polygon': [[line_cord[0], line_cord[1]], [line_cord[0] + line_cord[2], line_cord[1]],
                        [line_cord[0] + line_cord[2], line_cord[1] + line_cord[3]],
                        [line_cord[0], line_cord[1] + line_cord[3]]]
        })
    img_path = os.path.join(image_folder, filename)
    with Image.open(img_path) as img:
        img_width, img_height = img.size
        output_dict['img_size'] = [img_height, img_width]

    output_filename = os.path.splitext(filename)[0] + '.json'
    # Write ground truth data to JSON files
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)
