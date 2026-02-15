# BERT Sentiment Classification

Multi-class sentiment classification on 568,454 customer reviews using:

- TF-IDF + Logistic Regression (Classical ML)
- Fine-tuned BERT (Transformer Model)

Target Labels:
- Positive
- Neutral
- Negative

---

## 📊 Dataset

- Total samples: 568,454
- Highly imbalanced (Positive dominant)
- Classical models: ~80/20 train-test split
- BERT fine-tuning: 50,000 samples (due to computational constraints)

---

## 🔍 Methodology

### 1️⃣ Exploratory Data Analysis
- Class distribution analysis
- Text length distribution
- Word frequency patterns

---

### 2️⃣ TF-IDF + Logistic Regression

Three feature configurations were tested:

| Feature Set | Accuracy | Macro F1 |
|-------------|----------|----------|
| Summary Only | 0.89 | 0.71 |
| Review_Text Only | 0.87 | 0.63 |
| Summary + Review_Text | 0.76 | 0.31 |

**Observation:**  
- Summary-only performed best among classical models.  
- Neutral class was difficult to predict.  
- Combining both text fields increased sparsity and reduced performance.

---

### 3️⃣ Fine-Tuned BERT (50,000 samples)

- Pre-trained BERT model fine-tuned using PyTorch & HuggingFace Transformers
- Validation Accuracy: **0.897**
- Validation F1 Score: **0.893**

BERT achieved more balanced class prediction compared to classical ML models due to contextual embeddings.

---

## 📈 Model Comparison

| Model | Accuracy | Macro F1 |
|--------|----------|----------|
| Logistic Regression (Best) | 0.89 | 0.71 |
| Fine-Tuned BERT | 0.897 | 0.893 |

BERT significantly improved macro F1 score, especially for minority classes.

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- PyTorch
- HuggingFace Transformers
- Pandas / NumPy / Matplotlib

---




