import os
import logging
import numpy as np

from doc_ufcn_0_1_8.doc_ufcn.train import predict, evaluate, experiment, mlflow_utils
import configuration as config

loaders = experiment.prediction_loaders(config.norm_params, config.data_paths, config.input_size)
model = experiment.prediction_initialization(config.model_path_vf, config.classes_names, config.log_path)

if __name__ == '__main__':
    model = experiment.prediction_initialization(config.model_path_vf, config.classes_names, config.log_path)
    for threshold in [0.9]:  # np.linspace(0,1,11):
        print(threshold)
        predict.run(
            config.prediction_path,
            config.log_path,
            config.input_size,
            config.classes_colors,
            config.classes_names,
            config.save_image,
            config.min_cc,
            loaders,
            model,
            threshold
        )
        """logger = logging.getLogger(__name__)
        with mlflow_utils.start_mlflow_run(config.mlflow) as run:
            logger.info(f"Started MLflow run with ID ({run.info.run_id})")
        set = 'test'  # for set in config["data_paths"].keys():
        for dataset_path in config.data_paths[set]["json"]:
            if os.path.isdir(dataset_path):
                evaluate.run(
                    config.log_path,
                    config.classes_names,
                    set,
                    dataset_path,
                    str(dataset_path.parent.parent.name),
                    config.prediction_path,
                    config.evaluation_path,
                    config.mlflow_logging,
                )
            else:
                logging.info(f"{dataset_path} folder not found.")
        break"""
