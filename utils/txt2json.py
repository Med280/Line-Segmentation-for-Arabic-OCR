import os, json
from PIL import Image
from tqdm import tqdm
# Specify the file path
folder_txt = '/home/omar/Documents/newelyadata/out'
output_dir = "/home/omar/Documents/newelyadata/out_json"
images_dir = "/home/omar/Documents/out"


for filename in tqdm(os.listdir(folder_txt)):
    if filename.endswith('.txt'):
        output_dict = {
            'text_line': [],
            'img_size': []
        }
        with open(os.path.join(folder_txt, filename), 'r') as file:
            # Initialize variables for minimum and maximum values
            min_col1 = float('inf')
            min_col2 = float('inf')
            max_col3 = float('-inf')
            max_col4 = float('-inf')
            for line in file:
                # Split the line into individual values
                values = line.split()
                # Update the minimum and maximum values
                min_col1 = min(min_col1, int(values[0]))
                min_col2 = min(min_col2, int(values[1]))
                max_col3 = max(max_col3, int(values[2]))
                max_col4 = max(max_col4, int(values[3]))
        margin = 10
        output_dict['text_line'].append({
            'confidence': 1,
            'polygon': [[min_col1-margin, min_col2-margin], [max_col3+margin, min_col2-margin], [max_col3+margin, max_col4+margin],
                        [min_col1-margin, max_col4+margin]]
        })
        # Get image size for each image
        img_path = os.path.join(images_dir,
                                os.path.splitext(filename)[0][:-6] + ".jpg")  # I assume here that all files with .jpg ext
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            output_dict['img_size'] = [img_height, img_width]

        output_filename = os.path.splitext(filename)[0][:-6] + '.json'
        # Write ground truth data to JSON files
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
