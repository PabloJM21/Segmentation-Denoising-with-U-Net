# Segmentation & Denoising with U-Net

The goal of the first part of the exercise (segmentation) is to implement the U-Net architecture very similar to the original publication and then use it for its initial purpose, semantic segmentation.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Downloading the Data](#downloading-the-data)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Notebook](#running-the-notebook)
- [Project Details](#project-details)
  - [Data Preparation](#data-preparation)
  - [Data Exploration](#data-exploration)
  - [Implement PyTorch Dataset](#implement-pytorch-dataset)
  - [Implement the U-Net](#implement-the-u-net)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Training with Boundary Channel](#training-with-boundary-channel)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

In this project, we will implement the U-Net architecture for two main tasks: segmentation and denoising. The primary focus is on segmenting nuclei in fluorescence microscopy images.

## Directory Structure

```plaintext
Segmentation-Denoising-UNet/
│
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   └── Segmentation_Denoising_UNet.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── data_exploration.py
│   ├── dataset.py
│   ├── unet_model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── train_with_boundary.py
├── data/
│   ├── dsb2018/
│       ├── train/
│       ├── test/
├── results/
│   ├── logs/
│   ├── models/
│   ├── figures/
└── scripts/
    ├── download_data.sh
    └── download_data.py
```
## Setup

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Git

### Cloning the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/Segmentation-Denoising-UNet.git
cd Segmentation-Denoising-UNet
```
## Downloading the Data

To keep the repository lightweight, the data is not included. You can download the data by running the provided script.

```sh
chmod +x scripts/download_data.sh
./scripts/download_data.sh

```

## Installing Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## Running the Notebook
After downloading the data, you can start the Jupyter notebook:



```sh
jupyter notebook notebooks/Segmentation_Denoising_UNet.ipynb
```

## Project Details

### Data Preparation
The data preparation steps involve organizing and preprocessing the dataset for training and testing. This is handled in the data_preparation.py script in the src/ folder.

- Script: data_preparation.py

## Data Exploration
In this step, we explore the dataset to understand its structure and visualize some examples. This is done in the data_exploration.py script.

- Script: data_exploration.py

## Implement PyTorch Dataset
We implement a custom PyTorch dataset to handle data loading and preprocessing. This implementation can be found in the dataset.py script.

- Script: dataset.py

## Implement the U-Net
We implement the U-Net architecture as described in Ronneberger et al. The code for this is in the unet_model.py script.

- Script: unet_model.py

## Training
We define the training functions and train the model, plotting the results of loss and metrics. The training code is in the training.py script.

- Script: training.py

## Evaluation
We evaluate the model on the test data to assess its performance. This evaluation is done in the evaluation.py script.

- Script: evaluation.py

## Training with Boundary Channel
To avoid merges of touching nuclei, we add a boundary channel to the learning objective and retrain the model. This process is handled in the train_with_boundary.py script.

- Script: train_with_boundary.py

## Results
The results of the segmentation and denoising tasks will be stored in the results/ directory, including logs, models, and figures.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
- Kaggle for providing the dataset.
