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
