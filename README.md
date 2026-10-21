# Diabetic-Retinopathy-Detection
## 📌 Research Overview

Automated Diabetic Retinopathy (DR) screening often relies on deep, compute-heavy neural networks that are difficult to deploy on edge hardware in local clinics. This benchmark investigates whether modern parameter-efficient backbones can match or exceed heavier architectures across the 5 standard clinical DR severity stages:

* **Stage 0:** No DR
* **Stage 1:** Mild
* **Stage 2:** Moderate
* **Stage 3:** Severe
* **Stage 4:** Proliferative DR

---

## 🔬 Experimental Results

All transfer models were fine-tuned for 5 epochs using pretrained ImageNet weights, Adam optimization ($\alpha = 10^{-4}$), and Cross-Entropy loss on a stratified 80/20 train/test split (28,086 training images, 7,022 testing images).

### Experimental Results (Per-Class Precision / Recall / F1-Score)

| Model | Accuracy | Class 0 (P / R / F1) | Class 1 (P / R / F1) | Class 2 (P / R / F1) | Class 3 (P / R / F1) | Class 4 (P / R / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **82%** | 0.86 / 0.96 / 0.91 | 0.50 / 0.01 / 0.02 | 0.64 / 0.59 / 0.61 | 0.53 / 0.47 / 0.49 | **0.72 / 0.56 / 0.63** |
| **MobileNetV3** | **81%** | 0.86 / 0.95 / 0.91 | 0.47 / 0.02 / 0.03 | 0.60 / 0.59 / 0.59 | 0.52 / 0.49 / 0.51 | **0.69 / 0.62 / 0.65** |
| *ResNet18* | 80% | 0.84 / 0.97 / 0.90 | 1.00 / 0.00 / 0.00 | 0.60 / 0.52 / 0.56 | 0.68 / 0.15 / 0.25 | 0.81 / 0.50 / 0.62 |
| *VGG16* | 81% | 0.84 / 0.99 / 0.91 | 1.00 / 0.00 / 0.00 | 0.67 / 0.52 / 0.58 | 0.76 / 0.17 / 0.27 | 0.87 / 0.41 / 0.56 |

---

## 💡 Key Clinical Takeaways

1. **Lightweight Backbones Captured Rare Severe Disease Better:**  
   On **Class 4 (Proliferative DR)**, **MobileNetV3-Large achieved the highest sensitivity** (Recall: **0.62**, F1: **0.65**), outperforming deeper baselines like VGG16 (Recall: 0.41, F1: 0.56) despite having ~4% of the parameter count.
   
2. **Overall Accuracy Masked Minority Stage Degradation:**  
   While aggregate accuracy hovered between **80% and 82%**, performance on **Stage 1 (Mild DR)** collapsed across all models ($F_1 \le 0.03$). The heavy class skew (25,802 Class 0 images vs. 708 Class 4 images) demonstrates that standard cross-entropy loss favors majority classes, motivating cost-sensitive loss functions for clinical triage.

---

## 📓 Notebook Documentation

* `dr-efficientnet-b0.ipynb`: Full PyTorch implementation, data loading, augmentation pipeline, training loop, and 82% accuracy test evaluation for EfficientNet-B0.
* `dr-mobilenet-v3large.ipynb`: Full PyTorch implementation, training loop, and evaluation for MobileNetV3-Large (achieving top Class 4 sensitivity).

---

## 👨‍💻 Author & Supervision

* **Lead Student Researcher:** Abdallah Owaidi  
  * Computer Science Undergraduate, Al-Hussein Technical University (HTU)
  * Email: owaidiabdallah@gmail.com
* **Project Supervisor:** Eng. Yara Alharahsheh
