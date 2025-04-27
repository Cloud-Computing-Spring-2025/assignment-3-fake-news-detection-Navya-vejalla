# Fake News Detection using Spark MLlib

This project builds a simple machine learning pipeline using **Apache Spark MLlib** to classify news articles as **FAKE** or **REAL** based on their content.  
The pipeline covers loading the dataset, preprocessing text, extracting features, training a classifier, and evaluating performance.

---

## 📂 Dataset Used
- **Dataset**: `fake_news_sample.csv`
- **Size**: 500 articles (250 FAKE, 250 REAL)
- **Generated using**: Python Faker library
- **Fields**:
  - `id`: Unique article ID
  - `title`: Title of the news article
  - `text`: Full text/content of the article
  - `label`: Ground truth label (`FAKE` or `REAL`)

---

## 🛠 Brief Task Descriptions

| Task | Description | Output Saved |
|:----|:-------------|:-------------|
| **Task 1** | Load CSV file, explore the dataset (show first 5 rows, count articles, distinct labels). | `task1_output.csv` |
| **Task 2** | Preprocess text: lowercase conversion, tokenize text into words, remove stopwords. | `task2_output.csv` |
| **Task 3** | Feature extraction: Apply HashingTF and IDF for TF-IDF features, index labels. | `task3_output.csv` |
| **Task 4** | Train a Logistic Regression model, predict on test data (80/20 split). | `task4_output.csv` |
| **Task 5** | Evaluate the model using Accuracy and F1 Score. | `task5_output.csv` |

Each task saves intermediate results to a separate CSV file for easy review.

---

## 🚀 How to Run the Code

### 1. Setup Environment

Make sure you have the following installed:
- **Python 3.7+**
- **PySpark**: Install it using pip if not already installed:
  ```bash
  pip install pyspark