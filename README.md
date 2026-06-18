# 🎨 AI Art vs Human Art Detector

| Links |
|---------|
| 🚀 Live Demo: https://ai-art-detector-wafgn6ejm3zkn2unzplqtu.streamlit.app/#ai-art-vs-human-art-detector|
| 💻 Repository: https://github.com/yourusername/AI-Art-Detector |


## 📌 Project Overview

The AI Art vs Human Art Detector is a machine learning application designed to classify artwork images as either **AI-generated** or **human-created** using interpretable image features. The project focuses on combining feature engineering, model comparison, explainable AI techniques, and deployment into a complete end-to-end machine learning workflow.

Unlike deep learning approaches that rely on large computational resources, this project utilizes handcrafted image characteristics such as color statistics, entropy, and edge density to build an efficient and interpretable classification system.

---

## 🎯 Objectives

* Distinguish AI-generated artwork from human-created artwork.
* Explore interpretable image-based features for classification.
* Compare multiple machine learning algorithms.
* Explain model decisions using SHAP.
* Deploy the solution as an interactive web application.

---

## 📂 Dataset

Dataset Used:

`ai_vs_human_art_interpretable_numeric.csv`

The dataset contains engineered image features extracted from artwork images and corresponding labels indicating whether the artwork was AI-generated or human-created.

---

## 🔍 Features Used

The following image characteristics were used for classification:

| Feature      | Description                                            |
| ------------ | ------------------------------------------------------ |
| width        | Image width                                            |
| height       | Image height                                           |
| aspect_ratio | Width-to-height ratio                                  |
| pixel_count  | Total number of pixels                                 |
| mean_r       | Mean red channel intensity                             |
| mean_g       | Mean green channel intensity                           |
| mean_b       | Mean blue channel intensity                            |
| std_r        | Standard deviation of red channel                      |
| std_g        | Standard deviation of green channel                    |
| std_b        | Standard deviation of blue channel                     |
| entropy      | Image information complexity                           |
| edge_density | Edge concentration detected using Canny edge detection |

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing

* Data loading and inspection
* Missing value verification
* Feature preparation
* Label processing

### 2. Exploratory Data Analysis (EDA)

* Dataset overview
* Statistical summaries
* Feature distribution analysis

### 3. Data Splitting

* Train-Test Split
* Model evaluation on unseen data

### 4. Feature Scaling

* StandardScaler implementation

### 5. Model Training & Evaluation

The following algorithms were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

### 6. Model Selection

The best-performing model was selected based on evaluation performance.

🏆 **Best Model: Random Forest Classifier**

---

## 📊 Model Comparison

Multiple machine learning models were benchmarked to identify the most effective classifier for distinguishing AI-generated and human-created artwork.

Evaluation included:

* Accuracy comparison
* Model performance visualization
* Classification analysis

Random Forest achieved the strongest overall performance and was selected for deployment.

---

## 🔬 Explainable AI (SHAP)

To improve transparency and interpretability, SHAP (SHapley Additive exPlanations) was integrated into the workflow.

Implemented visualizations:

* SHAP Summary Plot
* SHAP Waterfall Plot
* Feature Importance Analysis

These explainability techniques help identify which image characteristics contribute most to model predictions.

---

## 📈 Evaluation Metrics

Model evaluation included:

* Confusion Matrix
* Feature Importance Analysis
* SHAP Explainability
* Prediction Confidence Scores

---

## 🚀 Streamlit Deployment

The trained Random Forest model was deployed using Streamlit, providing an interactive interface for real-time predictions.

### Application Features

✅ Image Upload Support

✅ Automatic Feature Extraction

✅ Real-Time Prediction

✅ Confidence Score Display

✅ User-Friendly Interface

Workflow:

Upload Image
→ Extract Features
→ Scale Features
→ Random Forest Prediction
→ Confidence Score

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* XGBoost
* CatBoost

### Explainability

* SHAP

### Image Processing

* OpenCV
* Pillow
* NumPy

### Deployment

* Streamlit

### Data Analysis

* Pandas

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Art-Detector.git
cd AI-Art-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## ▶️ Usage

1. Launch the Streamlit application.
2. Upload an artwork image.
3. The application extracts image features automatically.
4. Features are scaled using the trained StandardScaler.
5. The Random Forest model generates a prediction.
6. The confidence score is displayed to the user.

---

## 📁 Project Structure

```text
AI-Art-Detector/
│
├── app.py
├── random_forest.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
└── dataset/
    └── ai_vs_human_art_interpretable_numeric.csv
```

---

## 🔮 Future Improvements

* Deep Learning-based image classification
* CNN and Vision Transformer integration
* Larger and more diverse artwork datasets
* Enhanced explainability dashboards
* Multi-class artwork classification
* API deployment using FastAPI
* Performance optimization for large-scale inference

---

## 📚 Key Learning Outcomes

* Feature Engineering
* Model Comparison and Selection
* Explainable AI with SHAP
* Image Processing Techniques
* Machine Learning Deployment
* End-to-End ML Workflow Development

---

## 👨‍💻 Author

Developed as a machine learning portfolio project demonstrating data preprocessing, feature engineering, model evaluation, explainability, and deployment using modern ML tools and frameworks.
