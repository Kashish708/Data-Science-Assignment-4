# Data-Science-Assignment-4

# Learning Probability Density Functions using Data Only

---

## Objective

The objective of this assignment is to learn an unknown probability density function (PDF) of a transformed random variable using **only sample data**, without assuming any analytical or parametric distribution. A **Generative Adversarial Network (GAN)** is used to implicitly model the probability density of the transformed variable.

---

## Dataset Description

- **Dataset:** India Air Quality Data
- **Feature Used:** NO₂ concentration (x)
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

Only valid NO₂ values are used after removing missing entries.

---

## Data Transformation

Each NO₂ concentration value \( x \) is transformed into a new variable \( z \) using the following nonlinear transformation:

\[
z = x + a_r \sin(b_r x)
\]

Where:

- a_r = 0.05 ∗ (r mod 7)
- b_r = 0.3 ∗ (r mod 5 + 1)
- r is the university roll number

### Transformation Parameters

- **\( a_r \):** `3.0`
- **\( b_r \):** `0.6`

This transformation introduces nonlinearity in the data, making analytical PDF estimation difficult and suitable for a data-driven approach.

---

## Methodology

### Step 1: Data Preprocessing
- Extract NO₂ concentration values from the dataset
- Remove missing values
- Apply nonlinear transformation to obtain \( z \)
- Normalise transformed values for stable GAN training

---

### Step 2: GAN-Based PDF Estimation

A one-dimensional **Generative Adversarial Network (GAN)** is used to learn the unknown distribution of the transformed variable \( z \).

#### Generator Architecture
- Input: Noise sampled from standard normal distribution \( \mathcal{N}(0,1) \)
- Layers:
  - Dense layer (1 → 32) with ReLU activation
  - Dense layer (32 → 64) with ReLU activation
  - Dense layer (64 → 1)
- Output: Generated samples \( z_f \)

#### Discriminator Architecture
- Input: Real or generated samples
- Layers:
  - Dense layer (1 → 64) with LeakyReLU activation
  - Dense layer (64 → 32) with LeakyReLU activation
  - Dense layer (32 → 1) with Sigmoid activation
- Output: Probability of input sample being real

#### Training Configuration

| Parameter     | Value                |
|---------------|----------------------|
| Optimizer     | Adam                 |
| Learning Rate | 0.0002               |
| Loss Function | Binary Cross-Entropy |
| Batch Size    | 128                  |
| Epochs        | 3000                 |

The generator and discriminator are trained adversarially until the generated samples resemble the real data distribution.

---

### Step 3: PDF Approximation from Generator Samples

After training the GAN:
- A large number of samples are generated from the generator
- The probability density function is estimated using:
  - Histogram density estimation
  - Kernel Density Estimation (KDE)

---

## Results

### PDF Plot from GAN Samples

The following plot shows the estimated probability density function of the transformed variable \(z\) based on GAN-generated samples.

📌 **<img width="726" height="567" alt="image" src="https://github.com/user-attachments/assets/e8fdba55-1d20-4cae-86f1-0d6da0c787ba" />

**

---

### Result Summary

| Aspect               | Observation                       |
|----------------------|-----------------------------------|
| Mode Coverage        | Major modes successfully captured |
| Training Stability   | Stable after normalization        |
| Distribution Quality | Smooth and realistic PDF          |

---

## Observations

### Mode Coverage
The GAN captures the dominant modes of the transformed data distribution. With sufficient training, the generator produces samples covering high-density regions of the real data.

### Training Stability
Data normalisation and balanced learning rates ensure stable GAN training. Loss values converge without severe oscillations.

### Quality of Generated Distribution
The KDE curve obtained from generated samples closely matches the histogram density, indicating effective learning of the underlying probability distribution without assuming any parametric form.

---



