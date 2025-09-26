

# Airbus Ship Detection

A deep learning project developed for the [Airbus Ship Detection Challenge on Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection), aiming to detect and segment ships in satellite images — even under challenging conditions like clouds, haze, or small ship sizes.

## Project Overview

Maritime monitoring is crucial to prevent piracy, illegal fishing, smuggling, and environmental hazards. This project leverages **Convolutional Neural Networks (CNNs)** for ship segmentation in satellite imagery, supporting applications in **security, environmental protection, and maritime logistics**.

## Objectives

* Preprocess and clean large-scale satellite image datasets.
* Design and train a **CNN-based segmentation model** to accurately identify ships.
* Implement evaluation using **F2 Score at multiple IoU thresholds** to balance recall and precision.
* Explore fast inference approaches suitable for large-scale monitoring.

## Methods & Workflow

1. **Data Preprocessing**

   * Handled noisy and imbalanced data.
   * Applied image augmentation (rotations, flips, crops) to improve model generalization.

2. **Model Design**

   * Implemented a **U-Net / CNN-based segmentation model**.
   * Optimized hyperparameters for accuracy and speed.

3. **Evaluation**

   * Used **F2 Score with IoU thresholds (0.5–0.95)** as per Kaggle competition metric.
   * Assessed model agreement with ground-truth masks.

## Results

* Achieved **robust ship segmentation** across varied conditions (clouds, haze, different ship sizes).
* Delivered a model with strong performance (you can insert your exact metric/score here if available).

## Tools & Technologies

* **Languages:** Python
* **Libraries:** TensorFlow/Keras, OpenCV, NumPy, Pandas, Scikit-learn
* **Platform:** Kaggle (GPU acceleration)

## Repository Structure

```
├── notebook          # Jupyter notebooks for preprocessing & training
└── README.md           # Project documentation
```

##  How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/airbus-ship-detection.git
   cd airbus-ship-detection
   ```
2. Install dependencies
3. Run training

   ```

## Competition

* **Host:** Airbus (Kaggle)
* **Metric:** F2 Score @ IoU thresholds (0.5–0.95)
* **Participants:** 15,000+ competitors worldwide

## Future Work

* Optimize inference speed for real-time monitoring.
* Experiment with **Transformer-based vision models (e.g., SegFormer, Vision Transformers)**.
* Deploy on cloud-based platforms for large-scale maritime monitoring.

## References

* [Airbus Ship Detection Challenge on Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection)
* Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)

