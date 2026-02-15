# bert-sentiment-classification
🧠 Multi-class sentiment classification on 568k reviews using TF-IDF + Logistic Regression and fine-tuned BERT (50k samples).
📌 Overview
This project implements a large-scale multi-class sentiment classification pipeline on customer review data and compares:
TF-IDF + Logistic Regression (Classical ML)
Fine-Tuned BERT (Transformer Model)
Target labels:
Negative
Neutral
Positive

📂 Dataset
Total Dataset Size: 568,454 samples
Train/Test Split used for classical models (~80/20)
Test Set Size: 113,691 samples
Highly imbalanced (Positive dominant)
Due to computational constraints, 50,000 samples were used for BERT fine-tuning.

🔬 Methodology
1️⃣ Exploratory Data Analysis
Class distribution
Text length analysis
Word frequency patterns

Notebook: 1_EDA.ipynb
2️⃣ TF-IDF + Logistic Regression

Three feature configurations were tested:

Feature Set	                  Accuracy	        Macro F1
Summary Only	                   0.89	             0.71
Review_Text Only                 0.87	             0.63
Summary + Review_Text	           0.76	             0.31

Observations:
Summary-only performed best among classical models.
Neutral class remained difficult to predict.
Combining both text fields increased sparsity and reduced performance.

Notebook: 2_TFIDF_Logistic_Regression.ipynb
3️⃣ Fine-Tuned BERT (50,000 samples)
Pre-trained BERT model fine-tuned for multi-class classification using PyTorch and HuggingFace Transformers.
Final Performance (Validation)
Accuracy: 0.897
F1 Score: 0.893
BERT demonstrated more balanced class prediction compared to classical ML models due to contextual embeddings.

Notebook: 3_BERT_Sent.ipynb
📊 Model Comparison
Model	                             Accuracy       	Macro F1
Logistic Regression (Best)	       0.89	             0.71
Fine-Tuned BERT	                   0.897	           0.893

Although classical ML achieved competitive accuracy, BERT significantly improved balanced performance across minority classes.

🛠 Technologies Used

Python
Pandas
Scikit-learn
PyTorch
HuggingFace Transformers
Matplotlib




