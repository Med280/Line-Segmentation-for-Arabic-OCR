import os, json
from tqdm import tqdm
import math


def unnormalize_label_studio(bbox, width, height):
    return ((bbox[0] * width / 100),
            (bbox[1] * height / 100),
            (bbox[2] * width / 100),
            (bbox[3] * height / 100),)


output_dir = "/home/omar/Documents/70_one_article_json"  # /new elya data/annottaed-labels_json"
with open("/home/omar/Downloads/pre_annotation_for_new_one_article_only.json", "r") as file:
    json_file = json.load(file)
for i in tqdm(range(len(json_file))):
    dic = json_file[i]
    string = dic["data"]["ocr"]
    filename = string.split("/")[-1][9:]
    key = "predictions"
    if key == "predictions":
        for i, list in enumerate(dic['predictions']):
            output_dict = {
                'text_line': [],
                'img_size': []
            }
            image_width, image_height = list['result'][0]["original_width"], list['result'][0]["original_height"]
            img_bbox = []
            img_text = []
            for result in list['result']:
                if result['type'] == 'textarea':
                    x_n, y_n, w_n, h_n = result["value"]["x"], result["value"]["y"], result["value"]["width"], \
                        result["value"][
                            "height"]
                    x, y, w, h = unnormalize_label_studio([x_n, y_n, w_n, h_n], image_width, image_height)
                    output_dict['text_line'].append({
                        'confidence': 80808080,
                        'polygon': [[round(x), round(y)], [round(x + w), round(y)], [round(x + w), round(y + h)],
                                    [round(x), round(y + h)]]
                    })

            output_dict['img_size'] = [image_height, image_width]
            output_filename = os.path.splitext(os.path.basename(filename))[0] + '.json'
            # Write ground truth data to JSON files
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
    else:
        for i, list in enumerate(dic['annotations']):
            if list["completed_by"]["email"] == 'khouloudtouil19@gmail.com':
                # if filename == "26668_1.jpg":
                #  print(re)
                output_dict = {
                    'text_line': [],
                    'img_size': []
                }
                image_width, image_height = list['result'][0]["original_width"], list['result'][0]["original_height"]
                img_bbox = []
                img_text = []
                for result in list['result']:
                    if result['type'] == 'textarea':
                        x_n, y_n, w_n, h_n = result["value"]["x"], result["value"]["y"], result["value"]["width"], \
                            result["value"][
                                "height"]
                        x, y, w, h = unnormalize_label_studio([x_n, y_n, w_n, h_n], image_width, image_height)
                        rotation = result["value"]["rotation"]

                        # Convert rotation angle to radians
                        angle_rad = math.radians(rotation)

                        # Calculate the corner points of the rotated rectangle
                        cos_angle = math.cos(angle_rad)
                        sin_angle = math.sin(angle_rad)

                        corner_points = []
                        corner_points.append((x, y))  # Bottom left
                        corner_points.append((x, y + h))  # Top left
                        corner_points.append((x + w, y + h))  # Top right
                        corner_points.append((x + w, y))  # Bottom right

                        rotated_points = []
                        for point in corner_points:
                            rotated_x = round((point[0] - x) * cos_angle - (point[1] - y) * sin_angle + x)
                            rotated_y = round((point[0] - x) * sin_angle + (point[1] - y) * cos_angle + y)
                            rotated_points.append((rotated_x, rotated_y))

                        output_dict['text_line'].append({
                            'confidence': 1,
                            'polygon': [[*rotated_points[0], ], [*rotated_points[3], ],
                                        [*rotated_points[2], ], [*rotated_points[1], ]]
                        })
                        """
                        output_dict['text_line'].append({
                            'confidence': 1,
                            'polygon': [[round(x), round(y)], [round(x + w), round(y)], [round(x + w), round(y + h)],
                                        [round(x), round(y + h)]]
                        })
                        """
                output_dict['img_size'] = [image_height, image_width]

            output_filename = os.path.splitext(os.path.basename(filename))[0] + '.json'
            # Write ground truth data to JSON files
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
