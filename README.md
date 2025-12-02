# 📘 Text Classification Lab – Spam Detection & News Group Classification

This lab focuses on building and evaluating multiple machine learning models for text classification tasks. The project includes preprocessing, vectorization, model training, performance comparison, and confusion matrix visualizations.

---

## 🟦 1. Introduction

This lab demonstrates how machine learning can be applied to text classification problems such as:

- **SMS Spam Detection**
- **20 Newsgroups Topic Classification**

The workflow includes:
- Data cleaning  
- Text preprocessing  
- Vectorization using CountVectorizer or TF-IDF  
- Training and evaluating multiple models  

---

## 🟩 2. Dataset

Two datasets were used:

### **SMS Spam Dataset**
- Columns: `label` (ham/spam), `message`
- Task: classify each message as **ham** (normal) or **spam**

Example records:

| label | message |
|-------|---------|
| ham | Go until jurong point, crazy... |
| ham | Ok lar... Joking wif u oni... |
| spam | Free entry in 2 a wkly comp to win FA Cup… |

---

## 🟧 3. Preprocessing Steps

Applied text preprocessing includes:

- Lowercasing  
- Removing punctuation  
- Tokenization  
- Stopword removal  
- Lemmatization  
- Creating a new column: **cleaned_message**

Example:

| Original | Cleaned |
|----------|----------|
| "Go until jurong point, crazy.." | "go jurong point crazy available bugis great" |

---

## 🟨 4. Vectorization (CountVectorizer / TF-IDF)

Two text vectorization methods were used:

- **CountVectorizer** — converts text into token counts  
- **TF-IDF Vectorizer** — converts text into weighted frequency scores  

These numerical features are then used by machine learning models.

---

## 🟪 5. Models Used

```python
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": SVC(kernel='linear', C=1.0),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}
