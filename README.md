















































































































































### Introduction

To evaluate the performance of the doc-ufcn model on public and private datasets, we can fine-tune the model and run experiments using the latest version 0.1.9. We provide shell scripts and configuration files to facilitate the experiments.
### Training

To train the model, we first need to adjust the training parameters in the experiments_config file. We can then specify the steps to be run during the experiment, the data paths, and the restored model in the experiments.csv file. Once these files are set up, we can run the following command:

```shell

$ ./run_dla_experiment.sh -c experiments.csv -s false
```

### Generating label masks

To generate label masks for the input images, we can use the generate_mask() function from the utils.py file. By default, the function returns masks with the original dimensions of the input images. However, if we want to resize the masks, we can modify the function call to include the network_size argument:

```python

mask.generate_mask(
                gt_regions['img_size'][1], gt_regions['img_size'][0], network_size, label_polygons,
                {"text_line": (0, 0, 255)}, output_path, network_size
            )
```