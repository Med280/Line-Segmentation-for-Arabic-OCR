import os, json
from tqdm import tqdm

output_dir = "/home/omar/Documents/yarmouk_ocr/__non reviewed _Azure/gt"  # /new elya data/annottaed-labels_json"
with open("/home/omar/Downloads/project-31-at-2023-06-23-11-15-3964665d.json", "r") as file:
    json_file = json.load(file)
for i in tqdm(range(len(json_file))):
    dic = json_file[i]
    string = dic["file_upload"]
    filename = string[9:]
    key = "annotations"
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
                    pass
    else:
        for i, list in enumerate(dic['annotations']):
            # if list["completed_by"]["email"] == 'khouloudtouil19@gmail.com':
            gt_transcriptions = []
            img_bbox = []
            img_text = []
            for result in list['result']:
                if result['type'] == 'textarea':
                    gt_transcriptions.append(result["value"]["text"][0])

    output_filename = os.path.splitext(os.path.basename(filename))[0] + '.txt'
    # Write ground truth data to JSON files
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as file:
        # Iterate over the transcriptions and write each line to the file
        for transcription in gt_transcriptions:
            file.write(transcription + "\n")
