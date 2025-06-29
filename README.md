# 📰 Fake News Detection Using Machine Learning

A Machine Learning-based system that classifies news articles as **Fake** or **Real** using natural language processing (NLP) and traditional ML classifiers like Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest. It uses a labeled dataset from reliable news sources and fake news sites to build predictive models for text classification.

---

## 📌 Project Overview

The goal of this project is to:
- Detect misleading or fake news automatically using machine learning.
- Analyze the performance of multiple classification algorithms.
- Enable manual testing of new news articles.
- Lay the foundation for real-world deployment in web or mobile applications.

---

## 🧠 About ML Models Used

Four supervised learning models are implemented and evaluated:

| Model                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Logistic Regression | A linear model that estimates probabilities using the logistic function.    |
| Decision Tree       | A non-linear model that uses a tree-like structure for decision-making.     |
| Gradient Boosting   | An ensemble technique that builds models sequentially to correct errors.    |
| Random Forest       | A bagging ensemble of multiple decision trees to reduce overfitting.        |

All models use `TF-IDF` (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors.

---

## 🧰 Tools Used

- **Jupyter Notebook / Google Colab** – for coding and experimentation
- **scikit-learn** – for ML algorithms and evaluation
- **Pandas, NumPy** – for data manipulation and preprocessing
- **Matplotlib, Seaborn** – for data visualization
- **Regular Expressions** – for text cleaning

---

## 💻 Tech Stack and Library Versions

| Tool/Library       | Version         |
|--------------------|-----------------|
| Python             | 3.8+            |
| pandas             | 1.5+            |
| numpy              | 1.23+           |
| scikit-learn       | 1.1+            |
| matplotlib         | 3.6+            |
| seaborn            | 0.12+           |

---

## 🧾 Dataset Information

The dataset is composed of two CSV files:
- **Fake.csv** – Articles labeled as fake (class 0)
- **True.csv** – Articles labeled as real (class 1)

### Structure:
Each file contains the following columns:
- `title` – Headline of the article
- `text` – Full body text of the article
- `subject` – Topic category
- `date` – Publication date

> Source: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🏁 Results

| Model                | Accuracy  | Precision | Recall | F1 Score |
|---------------------|-----------|-----------|--------|----------|
| Logistic Regression | 98.78%    | 0.99      | 0.99   | 0.99     |
| Decision Tree       | 99.52%    | 0.99      | 0.99   | 0.99     |
| Gradient Boosting   | 99.44%    | 0.99      | 0.99   | 0.99     |
| Random Forest       | 98.69%    | 0.99      | 0.98   | 0.99     |

---

## 🛠 Hardware Requirements

- Minimum: 4 GB RAM, Dual-Core CPU
- Recommended: 8+ GB RAM for faster processing and training

---

## 💿 Software Requirements

- Python 3.8+
- pip or conda for package management
- Jupyter Notebook (optional but recommended)
- Any modern OS (Windows/Linux/macOS)

---

## 🚀 Model Deployment (Planned/Future Scope)

- 🔧 Deploy using Flask/Streamlit for real-time predictions
- 🖥️ Create a REST API for external application integration
- 🌐 Host on platforms like Heroku, Render, or AWS EC2

---

## 🔍 Manual Testing Example

```python
news = str(input("Enter a news article text: "))
manual_testing(news)
```

Predicts the result from all 4 models:
✅ Not A Fake News
❌ Fake News

## 📌 How to Run This Project
Clone the repository:
```
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```
Install dependencies:
```
pip install -r requirements.txt
```
Run the notebook:
```
jupyter notebook fake_news_detector.ipynb
```

## ✅ Additional Features
Custom text preprocessing function using regex

Manual news input for real-time classification

Supports model comparison through accuracy and classification report

Can be extended using Deep Learning models like LSTM or BERT

## 📄 License
This project is licensed under the MIT License.

## 🤝 Contributing
Pull requests and contributions are welcome. If you find bugs or want to add new features, feel free to fork the repo and create a PR.

## ✍️ Author
<p><strong>Deva the Analyst</strong><br>
Machine Learning & Data Science Enthusiast</p>
<p>
<a href="mailto:devasivanagasai@gmail.com">📧 Email</a><br>
<a href="https://www.linkedin.com/in/deva-siva-naga-sai-boddu-841984228/" target="_blank">🔗 LinkedIn</a> + <a href="https://github.com/Deva2078" target="_blank">💻 GitHub</a>
</p>

```
Let me know if you’d like me to create `requirements.txt`, `LICENSE`, `fake_news_detector.ipynb`, or deployment files as well.
```
