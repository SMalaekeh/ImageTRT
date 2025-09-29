
# ImageTRT

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Image Treatment Effect Estimation using **synthetic image generation** and deep learning.  
This project investigates how to estimate **direct and indirect treatment effects** when treatments are images, while accounting for **spatial spillover effects** and **aggregation bias**.

---

## 📂 Project Structure

- **`Synthetic Data Generation/`**  
  Generate synthetic datasets for treatments and outcomes (e.g., wetlands, DEM, capital, outcome, ITE, theta).

- **`Model/`**  
  - `main.ipynb`: Core notebook experimenting with different model architectures.  
  - Integrates utilities for embeddings, regressions, and evaluation.  

- **`Utils/`**  
  Utilities for:  
  - Image preprocessing (`load_and_resize.py`, `convolution.py`)  
  - Synthetic treatment/outcome generation  
  - Treatment & outcome regressions  
  - Wetland selection  
  - Evaluation functions for treatment effects  

---

## 🚀 Approach

We test combinations of:
- **Autoencoders**
- **Transformer Architectures**
- **CNN-based models**  
- Other embedding approaches  

The objective is to determine the most effective way to capture **direct** and **indirect** causal effects in high-dimensional spatial data.

---

## ✅ Current Progress
 
- Early model experiments in `Model/main.ipynb` (CNNs, autoencoders, embeddings)  
- Work-in-progress: spatial spillovers + aggregation bias integration  

