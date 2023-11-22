from pathlib import Path

from doc_ufcn_0_1_8.doc_ufcn import models

norm_params_tr_scratch = dict(mean=[196, 183, 159], std=[44, 43, 39])  # norm params of elyadata_ds
model_path_vf = "/home/omar/Videos/dep_model/Fine_tuning_DOCufcn_on_new_additional_dataset/last_model_0.pth"
model_path, parameters = models.download_model('generic-historical-line')
log_path = Path("/home/omar")
#log_path = Path("/home/omar/Documents/logs")
prediction_path = "predictions"
evaluation_path = "evaluation"
new_model_path = "fine_tuned model"
tb_path = "Tensorboard"
mean = "mean"
std = "std"

num_workers = 0
bin_size = 20
batch_size = 1
use_amp = False
learning_rate = 5e-3
no_of_epochs = 4
training = {
    "restore_model": model_path,
    "loss": "initial"  # loss will be initialized if the dataset isn't seen on training
}

norm_params = dict(mean=parameters["mean"], std=parameters["std"])
classes_names = parameters["classes"]
input_size = parameters["input_size"]
min_cc = parameters["min_cc"]
classes_colors = [[0, 0, 0], [0, 0, 255]]
save_image = [""]

model_path = model_path
data_paths = {
    "train": {
        "image": [Path("/home/omar/Videos/images/example")],
        "mask": [Path("/home/omar/Documents/trial2/train/labels")],
        "json": [Path("/home/omar/Documents/elyadata_ds/train/labels_json")],
    },
    "val": {
        "image": [Path("/home/omar/Videos/images/example")],
        "mask": [Path("/home/omar/Documents/trial2/val/labels")],
        "json": [Path("/home/omar/Documents/elyadata_ds/val/labels_json")],
    },
    "test": {
        "image": [Path("/home/omar/Documents/elyadata_ds_p_n_g/test/images")],
        "json": [Path("/home/omar/Documents/elyadata_ds_p_n_g/test/labels_json")],
    },
}
mlflow_logging = False
mlflow = {
    "experiment_id": 442826864979463009,  # 29
    "run_name": "test run",
    "tracking_uri": "http://127.0.0.1:5000",  # http://192.168.1.7:15000/
    "s3_endpoint_url": "null"
}
