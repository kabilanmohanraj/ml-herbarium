## For usage on SCC, first, create a conda environment using the following:
```
module load miniconda
```
```
conda create -n my_env -c conda-forge python=’version’
```
```
conda activate my_env
```

## The 'sec_task3.ipynb' file is used to download the images for the primary task dataset by filtering the GBIF data based on the New England dataset.
Run the cells in this notebook to download the images and save it in the desired path.

## The 'Mask_remove_text-3.ipynb' file is used to remove the labels from the image so that there is no bias influencing the classifier.
Add the following modules first(after creating a conda nevironment with python version > 3.9): 
```
module load tensorflow/2.11.0
```
```
module load gcc/12.2.0
```
```
module load bazel/4.1.0
```
```
module load cuda/11.2
```
```
module load cudnn/8.1.1
```
(conda install module_name – for all other packages)
Run the cells in this notebook to create a datset of the images with masked labels.

## 'IELT_data_prep.ipynb' is just used to convert the downloaded data to the format required by the IELT model(.tgz format)

To run the IELT model follow the following instructions:
## !! NOTE THAT THIS MODEL HAS BEEN SETUP TO RUN ON THEIR CUB DATASET, NOT ON THE DOWNLOADED HERBARIUM DATASET !!
## Requirements

python     >= 3.9

pytorch	>= 1.8.1
```
conda install -c conda-forge tqdm
```
```
conda install -c conda-forge timm
```

## Training

1. Put the pre-trained ViT model in `pretrained/`, and rename it to `ViT-B_16.npz`, you can download from [ViT pretrained](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz).
2. Select a experiments setting file in `configs/`, and Modify the path of `dataset`.
3. Modify the path in `setup.py` in line 5, and you can change the log name and cuda visible by modify line 13,14.
4. Running the following code according to you pytorch version:

### Single GPU

```bash
python -m main.py
```

### Multiple GPUs

#### If pytorch < 1.12.0

```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py 
```

#### If pytorch >= 1.12.0

```
torchrun --nproc_per_node 4 main.py
```

You need to change the number behind the `-nproc_per_node` to your number of GPUs


## We ran and evaluated the code of the IELT model on their cub dataset
We have trained the model on cub dataset. We uploaded a downloaded version of the dataset as the download function in the code isn't working properly, it's throwing ambiguous errors. While training the model on a particular dataset it is important to remember that we need to change the directory path for the dataset in both the 'base.yaml' and the respective 'datset.yaml' file(in this case cub.yaml).
The following image shows the performance on the cub dataset:

![WhatsApp Image 2023-05-11 at 9 02 54 PM](https://github.com/kabilanmohanraj/ml-herbarium/assets/62757336/8860936a-d505-40e4-80c4-df40af1e3c76)
