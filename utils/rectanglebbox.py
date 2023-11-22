import os
import json
import cv2
import numpy as np

# Set the input and output directories
input_dir = '/home/omar/Videos/labels_json'
output_dir = '/home/omar/Videos/rect_labels_json'

# Loop through all XML files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        output_dict = {
            'text_line': [],
            'img_size': []
        }
        with open(os.path.join(input_dir, filename), "r") as file:
            gt_regions = json.load(file)
            for dic in gt_regions['text_line']:
                gt_points = dic['polygon']
                x, y, w, h = cv2.boundingRect(np.array(gt_points))
                output_dict['text_line'].append({
                    'confidence': 1,
                    'polygon': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                })
                output_dict['img_size'] = [gt_regions['img_size'][0], gt_regions['img_size'][1]]

        # output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
