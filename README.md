# Deepfake Video Detection

This project aims to detect deepfake videos by using a machine learning model. It consists of two main components:
1. **Feature Extraction** - Extracts features from video frames.
2. **Classification Model** - Classifies videos as real or fake using a simple fully connected neural network.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

---

## Project Overview
This project focuses on detecting deepfake videos by extracting meaningful features from video frames using a pre-trained ResNet model, followed by training a neural network to classify the videos as real or fake.

---

## Dependencies
The project relies on the following libraries:
- `torch` - For building and training the deep learning model.
- `torchvision` - To utilize pre-trained ResNet for feature extraction.
- `opencv-python` - To process video frames.
- `numpy` - For numerical operations and handling data.
- `scikit-learn` - For metrics (e.g., EER) and dimensionality reduction.
- `tqdm` - For progress bar display during training.
- `matplotlib` - To visualize sample frames.

Install dependencies:
```bash
pip install torch torchvision opencv-python-headless numpy scikit-learn tqdm matplotlib
