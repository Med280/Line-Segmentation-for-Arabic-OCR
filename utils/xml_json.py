import os
from xml.dom import minidom
import json

# Set the input and output directories
input_dir = '/home/omar/Desktop/RASM/RASM_2/labels_xml'
output_dir = '/home/omar/Desktop/RASM/RASM_2/labels_json'

# Loop through all XML files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        # Load the XML file
        xml_path = os.path.join(input_dir, filename)
        xmldoc = minidom.parse(xml_path)

        # Extract the relevant information
        textline_tags = xmldoc.getElementsByTagName('TextLine')
        output_dict = {
            'text_line': [],
            'img_size': []
        }
        for textline_tag in textline_tags:
            coords_tag = textline_tag.getElementsByTagName('Coords')[0]
            points_str = coords_tag.getAttribute('points')
            points = []
            for point_str in points_str.split():
                x, y = point_str.split(',')
                points.append([int(x), int(y)])
            output_dict['text_line'].append({
                'confidence': 1,
                'polygon': points
            })

        # Get the image size from the Page tag
        page_tag = xmldoc.getElementsByTagName('Page')[0]
        img_width = int(page_tag.getAttribute('imageWidth'))
        img_height = int(page_tag.getAttribute('imageHeight'))
        output_dict['img_size'] = [img_height, img_width]

        # Save the output as a JSON file
        output_filename = os.path.splitext(filename)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=4)
