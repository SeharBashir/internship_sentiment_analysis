# 🧠 Internship Feedback Sentiment Analysis

This project performs **sentiment analysis** on internship feedback using **DistilBERT (Hugging Face Transformers)** in a **single Python script**.
It classifies feedback as **Positive** or **Negative**, helping organizations analyze intern experiences efficiently.

---

## 🎯 Objectives

* Fine-tune a **pretrained Transformer model (DistilBERT)** for sentiment analysis.
* Analyze and classify feedback into **positive** or **negative** sentiments.
* Evaluate the model’s performance and visualize results.

---

## 🧰 Tech Stack

| Category        | Tools / Libraries                      |
| --------------- | -------------------------------------- |
| Programming     | Python 3                               |
| IDE             | VS Code                                |
| NLP Model       | DistilBERT (Hugging Face Transformers) |
| Deep Learning   | PyTorch                                |
| Data Handling   | Pandas, NumPy                          |
| Evaluation      | Scikit-learn                           |
| Visualization   | Matplotlib, Seaborn                    |
| Version Control | Git + GitHub                           |

---

## 📁 Project Structure

```
internship_sentiment_analysis/
│
├── data/
│   └── feedback.csv                # dataset (your 30+ feedback rows)
│
├── sentiment_analysis.py           # main script (training + prediction)
├── requirements.txt                # dependencies list
└── README.md                       # project documentation
```

---

## ⚙️ How to Run on VS Code

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SeharBashir/internship_sentiment_analysis.git
cd internship_sentiment_analysis
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # (on Windows)
# or
source venv/bin/activate  # (on macOS/Linux)
```

### 3️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Model

```bash
python sentiment_analysis.py
```

> 💡 This script automatically loads **DistilBERT** from Hugging Face and fine-tunes it on your dataset (`data/feedback.csv`).

---

## 📊 Example Output

```
Training Accuracy: 0.94
Validation Accuracy: 0.91
Classification Report:
               precision    recall  f1-score   support
    negative       0.90      0.92      0.91        15
    positive       0.93      0.91      0.92        15
    accuracy                           0.91        30
```

---

## 💡 Insights

* Positive feedback often includes *learning*, *teamwork*, and *support*.
* Negative feedback tends to focus on *workload*, *communication*, and *organization*.
* The Transformer model performs better than traditional ML models like Naive Bayes.

---

## 🚀 Future Enhancements

* Add **neutral** category for more nuanced classification.
* Create a **Streamlit dashboard** to visualize results.
* Extend to **aspect-based sentiment** (mentor, workload, environment).

---

## 👩‍💻 Author

**Sehar Bashir**


## 📝 License

Open-source under the [MIT License](LICENSE).
