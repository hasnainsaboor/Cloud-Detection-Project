# Cloud-Detection-Project
An AI-based cloud detection and masking system for satellite imagery using Convolutional Neural Networks (CNNs). Semester 4 SPM Project.
## 📝 Project Description
Satellite optical imagery is frequently obscured by clouds, which reduces the usable area of an image and complicates downstream tasks such as land-use monitoring, disaster response, and agricultural analysis. Analysts currently rely heavily on slow and inconsistent manual inspection or basic thresholding methods.

This project is inspired by real-world remote-sensing research needs (SUPARCO RESOLVE South R&D Project #25). It provides a machine-learning-based system using Convolutional Neural Networks (CNNs) to automatically detect and mask clouds in satellite images at the pixel level. 

**Target Objective:** Achieve a segmentation accuracy of at least 85% (Intersection-over-Union) on a held-out test set.

 📂 Repository Structure

├── data/                   # Dataset folder (ignored in git)
│   ├── train/              # Training images and masks
│   └── test/               # Testing images and masks
├── notebooks/              # Jupyter Notebooks / Google Colab experiments
├── src/                    # Core source code
│   ├── data_prep.py        # Data loading and augmentation scripts
│   ├── model.py            # CNN / U-Net architecture definition
│   └── train.py            # Training loops and evaluation metrics
├── app.py                  # Streamlit / Gradio User Interface
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
