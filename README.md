# MNIST Dataset Loading and Preparation

## 📌 Overview

This project uses the **MNIST handwritten digit dataset** to work with image classification.

The MNIST dataset contains grayscale images of handwritten digits from **0 to 9**. Each image has a resolution of **28 × 28 pixels**, giving a total of **784 pixel values** per image.

The dataset is downloaded from Kaggle using `kagglehub` and loaded into a Pandas DataFrame.

---

## 📂 Dataset Download

The dataset is downloaded using the KaggleHub Python library:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

print("Path to dataset files:", path)
```

After downloading, the files inside the dataset directory can be checked using:

```python
import os

print(os.listdir(path))
```

The downloaded dataset contains:

```text
mnist_train.csv
mnist_test.csv
```

---

## 📊 Loading the Training Dataset

The training CSV file is loaded using Pandas:

```python
import pandas as pd
import os

data = pd.read_csv(os.path.join(path, "mnist_train.csv"))
```

The resulting DataFrame contains the digit label and the pixel values representing each handwritten digit.

---

## 🧩 Dataset Structure

Each MNIST image is:

```text
28 × 28 pixels = 784 pixels
```

Therefore, one row of the CSV can be viewed conceptually as:

```text
┌─────────┬────────┬────────┬────────┬───────┬──────────┐
│  label  │ pixel0 │ pixel1 │ pixel2 │  ...  │ pixel783 │
├─────────┼────────┼────────┼────────┼───────┼──────────┤
│    7    │   0    │   0    │   0    │  ...  │    0     │
└─────────┴────────┴────────┴────────┴───────┴──────────┘
```

### Label

The `label` represents the actual digit in the image.

For example:

```text
0 → Digit 0
1 → Digit 1
2 → Digit 2
...
9 → Digit 9
```

### Pixel Values

The remaining columns contain pixel intensity values.

For a grayscale MNIST image:

```text
0   → Black
255 → White
```

Values between 0 and 255 represent different shades of gray.

---

## 🎯 Training Data

For this project, **5,000 samples** are used for training.

The data is divided into:

```text
                5,000 MNIST Samples
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          X_train              y_train
        Pixel Data              Labels
             │                   │
             ▼                   ▼
       5,000 × 784            5,000
       pixel values          digit labels
                              (0–9)
```

### `X_train`

`X_train` contains the input features.

Each image is represented by **784 pixel values**:

```text
28 × 28 = 784
```

Therefore:

```text
X_train.shape = (5000, 784)
```

### `y_train`

`y_train` contains the correct digit for each image.

Therefore:

```text
y_train.shape = (5000,)
```

For example:

```text
X_train[0] → Image of handwritten digit
y_train[0] → 7
```

The model uses `X_train` to learn the image patterns and `y_train` as the correct answer.

---

## 🔄 Complete Data Flow

```text
Kaggle MNIST Dataset
        │
        ▼
kagglehub.dataset_download()
        │
        ▼
mnist_train.csv
        │
        ▼
Pandas DataFrame
        │
        ▼
Select 5,000 samples
        │
        ├──────────────────┐
        ▼                  ▼
     X_train             y_train
   5,000 × 784           5,000
    pixel values         labels
        │                  │
        └────────┬─────────┘
                 ▼
          Model Training
                 │
                 ▼
       Learn handwritten patterns
                 │
                 ▼
          Predict digit 0–9
```

---

## 🛠️ Libraries Used

* **KaggleHub** – Download the dataset from Kaggle
* **Pandas** – Load and manipulate CSV data
* **NumPy** – Numerical operations on image/pixel data
* **Matplotlib** – Visualize handwritten digits
* **OS** – Work with dataset file paths

```python
import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## 🚀 Objective

The main objective is to prepare the MNIST handwritten digit data for a machine learning/deep learning classification model.

The workflow is:

**Download → Load → Explore → Prepare → Train → Evaluate → Predict**

The final model should be able to take a handwritten digit image as input and correctly classify it as one of the digits **0–9**.
