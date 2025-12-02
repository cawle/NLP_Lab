# Text Classification Lab – Spam Detection & News Group Classification

This lab focuses on building and evaluating multiple machine learning models for text classification tasks. The project includes preprocessing, vectorization, model training, performance comparison, and confusion matrix visualizations.

---

## 1. Introduction

This lab applies machine learning techniques to classify text data for two main tasks:

- **SMS Spam Detection**
- **20 Newsgroups Topic Classification**

The workflow includes:

- Data cleaning  
- Text preprocessing  
- Tokenization & stopword removal  
- Vectorization using CountVectorizer or TF-IDF  
- Training and evaluating multiple ML models  

---

## 2. Dataset

### **SMS Spam Dataset**
- Columns: `label` (ham/spam), `message`
- Goal: classify each message as **ham** (legit) or **spam**

**Sample:**

| label | message |
|-------|---------|
| ham | Go until jurong point, crazy... |
| ham | Ok lar... Joking wif u oni... |
| spam | Free entry in 2 a wkly comp to win FA Cup… |

---

## 3. Preprocessing Steps

Text preprocessing performed:

- Lowercasing  
- Removing punctuation  
- Tokenization  
- Stopword removal  
- Lemmatization  
- Creating a new feature: **cleaned_message**

**Example transformation:**

| Original | Cleaned |
|----------|----------|
| "Go until jurong point, crazy.." | "go jurong point crazy available bugis great" |

---

## 4. Vectorization

Two vectorization techniques were used:

### **CountVectorizer**
- Converts text into token count vectors.

### **TF-IDF Vectorizer**
- Computes weighted term frequency representation.
- Reduces importance of overly common words.

Both techniques were used to compare model performance.

---

## 5. Models Used

The following machine learning models were evaluated:

```python
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": SVC(kernel='linear', C=1.0),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}



## 6. Training & Evaluation

For each model, the following steps were performed:

1. Fit the model on the training data  
2. Predict results on the test data  
3. Compute accuracy  
4. Generate a classification report  
5. Plot the confusion matrix  

### **Metrics Used**
- Accuracy  
- Precision  
- Recall  
- F1-score  


---

## 7. Results & Accuracy

### 📌 Model Accuracy Summary (SMS Spam Detection)

| Model | Accuracy |
|--------|----------|
| Naive Bayes | **0.9784** |
| SVM (Linear) | 0.9754 |
| Logistic Regression | 0.9754 |
| Random Forest | 0.9706 |

➡️ **Naive Bayes achieved the highest accuracy in spam detection.**


---

## 8. Confusion Matrices

### **SMS Spam Detection – Naive Bayes**

|                 | Predicted Ham | Predicted Spam |
|-----------------|---------------|----------------|
| **Ham**         | 1438          | 15             |
| **Spam**        | 21            | 198            |

Heatmaps were generated to visualize correct vs. incorrect predictions.

---

### **20 Newsgroups Multi-Class Classification**

Confusion matrices were generated for the following categories:

- `comp.sys.mac.hardware`
- `rec.sport.baseball`
- `sci.crypt`
- `talk.politics.misc`

These visualizations show how often each category was predicted correctly or misclassified by the model.


---

## 9. Observations & Conclusions

- **Naive Bayes** performs exceptionally well for spam classification because it handles word-frequency data effectively.  
- **Linear SVM** delivers strong performance for both binary and multi-class classification tasks.  
- **Logistic Regression** remains accurate and stable across experiments.  
- **Random Forest** performs adequately but is less suitable for high-dimensional sparse text data such as TF-IDF or bag-of-words vectors.  
- Text preprocessing (lowercasing, stopword removal, lemmatization) significantly improves overall model accuracy.  
- **TF-IDF vectorization** generally outperforms basic token count vectorization.  
- Confusion matrices show minimal misclassifications, confirming high model reliability.


---
