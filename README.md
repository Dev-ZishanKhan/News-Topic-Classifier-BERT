# 📰 News Topic Classifier Using BERT

> Fine-tuned BERT model to classify news headlines into topic categories using the AG News Dataset, deployed as an interactive web app with Streamlit.

## 🌐 Live Demo

👉 **[Click here to try the classifier live!](https://proc-differ-upon-william.trycloudflare.com)**

---

## 📌 Objective

Build a news topic classifier that:
- Fine-tunes **bert-base-uncased** on AG News Dataset
- Classifies headlines into 4 categories: **World, Sports, Business, Sci/Tech**
- Evaluates using **Accuracy and F1-Score**
- Deploys as an interactive web app using **Streamlit**

---

## 🏗️ Methodology / Approach

### 1. 📊 Dataset
- Used **AG News Dataset** from Hugging Face
- **120,000 train** and **7,600 test** samples
- 4 balanced categories — 30,000 samples each
- Used **4,000 train** and **800 test** samples for fine-tuning

### 2. ⚙️ Tokenization & Preprocessing
- Used `BertTokenizer` from Hugging Face Transformers
- `max_length = 128`, padding and truncation applied
- Converted to PyTorch tensors for training

### 3. 🤖 Model Fine-Tuning
- Base model: `bert-base-uncased`
- Added classification head for 4 classes
- Trained for **3 epochs** with batch size 16
- Used `Trainer` API from Hugging Face

### 4. 📈 Evaluation
- Evaluated using **Accuracy** and **F1-Score**
- Generated **Confusion Matrix** and per-class F1 scores
- Visualized results with Matplotlib

### 5. 🌐 Deployment
- Deployed using **Streamlit** as interactive web UI
- Shows confidence scores for all 4 categories
- Publicly accessible via **Cloudflare Tunnel**

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| 🤖 Model | BERT (bert-base-uncased) |
| 🔗 Framework | Hugging Face Transformers |
| 📊 Dataset | AG News (Hugging Face) |
| 📈 Evaluation | Scikit-learn |
| 🌐 Deployment | Streamlit + Cloudflare |
| 💻 Platform | Google Colab (GPU) |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| ✅ Overall Accuracy | 90.6% |
| ✅ Macro F1 Score | 90.0% |
| 🥇 Best Category | Sports (F1: 0.967) |
| 📉 Hardest Category | Sci/Tech (F1: 0.865) |

---

## 🧪 Per Category Results

| Category | Precision | Recall | F1 Score |
|---|---|---|---|
| 🌍 World | 0.93 | 0.86 | 0.89 |
| ⚽ Sports | 0.95 | 0.99 | **0.97** |
| 💼 Business | 0.88 | 0.85 | 0.87 |
| 💻 Sci/Tech | 0.84 | 0.90 | 0.87 |

---

## 💡 Key Insights

1. **BERT achieved 90%+ accuracy** with only 4,000 training samples
2. **Sports** category easiest to classify — very distinct vocabulary
3. **Business & Sci/Tech** sometimes confused — overlapping topics
4. **Transfer learning** is powerful — minimal fine-tuning needed
5. **3 epochs** sufficient for convergence — no overfitting

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Dev-ZishanKhan/News-Topic-Classifier-BERT
cd News-Topic-Classifier-BERT
```

### 2. Install Dependencies
```bash
pip install transformers datasets torch scikit-learn streamlit
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
News-Topic-Classifier-BERT/
│
├── news_classifier_bert.ipynb   # Main Jupyter Notebook
├── app.py                       # Streamlit web application
├── preview                      # Images
└── README.md                    # Project documentation
```

---

## 📝 License

This project is part of the **DevelopersHub Corporation AI/ML Engineering Internship** — Advanced Task 1.

---

## 👨‍💻 Author

**Zishan Khan**
- 🌐 GitHub: [@Dev-ZishanKhan](https://github.com/Dev-ZishanKhan)
- 💼 LinkedIn: [Zishan-Coderx](https://www.linkedin.com/in/zishan-ai-engineer/)