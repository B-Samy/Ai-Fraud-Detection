# 💳 AI Fraud Detection

An end-to-end **Machine Learning fraud detection system** that analyzes financial transactions and predicts whether a transaction is **Fraudulent or Legitimate**.

The project covers the complete machine learning workflow — from **data exploration and preprocessing to model training, evaluation, and deployment using Streamlit**.

---

## 🚀 Project Overview

Financial fraud can cause significant losses for businesses and customers. Detecting fraudulent transactions automatically allows financial systems to identify suspicious activity and reduce potential losses.

This project uses machine learning classification algorithms to learn patterns from historical transaction data and predict whether a new transaction is:

* 🔴 **Fraudulent**
* 🟢 **Legitimate**

The project was designed as a practical demonstration of how an ML model can be integrated into a usable application.

---

## 🎯 Objectives

* Analyze transaction data and identify fraud patterns.
* Perform exploratory data analysis (EDA).
* Clean and preprocess transaction data.
* Engineer relevant features for fraud detection.
* Train multiple classification models.
* Compare model performance.
* Evaluate predictions using classification metrics.
* Save the trained model using `joblib`.
* Build an interactive Streamlit application.
* Deploy the fraud detection system as a web application.

---

## 🏗️ Machine Learning Workflow

```text
Transaction Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Encoding Categorical Features
        ↓
Feature Scaling
        ↓
Train / Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Save Model with Joblib
        ↓
Streamlit Application
        ↓
Fraud / Legitimate Prediction
```

---

## 📊 Dataset

The dataset contains transaction-level information used to identify potentially fraudulent transactions.

Typical transaction information includes:

* Transaction type
* Transaction amount
* Account balances
* Sender information
* Receiver information
* Fraud indicator

The target variable represents whether a transaction is fraudulent.

```text
0 → Legitimate
1 → Fraudulent
```

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand the structure and behavior of the transaction data.

Key analysis included:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Class distribution
* Transaction type distribution
* Transaction amount analysis
* Correlation analysis
* Fraud vs legitimate transaction patterns

Visualization tools such as **Matplotlib** and **Seaborn** were used to identify important patterns.

---

## 🧹 Data Preprocessing

The preprocessing pipeline included:

### 1. Missing Value Analysis

Checked the dataset for missing values and handled them appropriately.

### 2. Duplicate Analysis

Duplicate records were identified and removed where necessary.

### 3. Categorical Encoding

Categorical transaction information was converted into numerical form so that machine learning algorithms could process it.

### 4. Feature Selection

Relevant transaction features were selected for model training.

For example:

```text
Amount
Old Balance
New Balance
Transaction Type
```

Identifier fields such as sender/receiver names or IDs can be excluded when they do not provide meaningful predictive information.

### 5. Feature Scaling

Numerical features were scaled when required by the selected machine learning algorithm.

---

## 🤖 Machine Learning Models

Multiple classification algorithms can be evaluated to determine which approach performs best for fraud detection.

Models explored in the project include:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* AdaBoost

The models were compared using appropriate classification metrics rather than relying only on accuracy.

---

## 📈 Model Evaluation

Fraud detection is an **imbalanced classification problem**, meaning legitimate transactions can greatly outnumber fraudulent ones.

Therefore, evaluation focuses on metrics such as:

### Accuracy

Overall percentage of correct predictions.

### Precision

Of the transactions predicted as fraud, how many were actually fraudulent?

### Recall

Of all actual fraudulent transactions, how many did the model successfully detect?

### F1-Score

Balances precision and recall.

### Confusion Matrix

Shows:

```text
                    Predicted
                 Legit    Fraud

Actual Legit      TN        FP
Actual Fraud      FN        TP
```

For fraud detection, **recall is particularly important**, because missing an actual fraudulent transaction can be costly.

---

## 💾 Model Persistence

After selecting the trained model, it can be saved using `joblib`.

```python
import joblib

joblib.dump(model, "fraud_detection_model.pkl")
```

The saved model can then be loaded inside the Streamlit application:

```python
model = joblib.load("fraud_detection_model.pkl")
```

This allows the application to make predictions without retraining the model every time.

---

## 🌐 Streamlit Application

The trained ML model is integrated into a Streamlit interface.

Users can enter transaction information and receive a prediction.

### Example

```text
Transaction Details
        ↓
User Input
        ↓
Preprocessing
        ↓
ML Model
        ↓
Prediction
        ↓
Fraudulent / Legitimate
```

Example output:

```text
🟢 Transaction appears legitimate
```

or

```text
🔴 Potential fraudulent transaction detected
```

---

## 🧰 Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data manipulation         |
| NumPy        | Numerical computation     |
| Matplotlib   | Data visualization        |
| Seaborn      | Statistical visualization |
| Scikit-learn | Machine learning          |
| Joblib       | Model persistence         |
| Streamlit    | Web application           |
| Git & GitHub | Version control           |

---

## 📁 Project Structure

```text
AI-Fraud-Detection/
│
├── data/
│   └── fraud_dataset.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   └── fraud_detection_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔐 Security Considerations

The project demonstrates fraud detection using historical transaction data and should not be treated as a production financial security system without further validation.

A production system would require:

* Secure data handling
* Model monitoring
* Fraud investigation workflows
* Continuous retraining
* Data drift detection
* Explainable predictions
* Strong authentication and authorization
* Protection of sensitive financial information

---

## 🔮 Future Improvements

Potential improvements include:

* ⚖️ Advanced handling of class imbalance
* 🎯 Hyperparameter optimization
* 🔄 Cross-validation
* 📊 ROC-AUC and Precision-Recall analysis
* 🧠 Explainable AI using SHAP
* 🚨 Real-time transaction monitoring
* 📈 Model performance monitoring
* 🔁 Automated model retraining
* 🗄️ Database integration
* 🔐 Authentication and user management
* ☁️ Production cloud deployment

---

## 💡 Key Learning Outcomes

Through this project, I worked with:

* End-to-end ML pipelines
* Exploratory Data Analysis
* Data preprocessing
* Categorical encoding
* Feature engineering
* Feature selection
* Classification algorithms
* Imbalanced classification
* Model evaluation
* Confusion matrices
* Model persistence with Joblib
* Streamlit deployment
* Git/GitHub workflow

---

## 👨‍💻 Project Highlights

This project demonstrates the ability to take a machine learning problem from:

**Raw transaction data → preprocessing → model training → evaluation → saved model → deployed application**

rather than stopping at model training inside a notebook.

---

## 📌 Conclusion

The **AI Fraud Detection** project demonstrates an end-to-end approach to building a machine learning classification system for identifying potentially fraudulent financial transactions.

It combines **data science, machine learning, model evaluation, and application deployment** into a single practical project.

**Built with Python, Scikit-learn, Joblib, and Streamlit.**
