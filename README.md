# 🍅 Tomato Leaf Disease Classification & Severity Estimation

## 📌 Project Overview
Tomato crops are crucial to global food security, yet their yields are constantly threatened by bacterial, fungal, and viral diseases affecting leaf tissue. Early and accurate detection is essential to minimise yield loss and guide effective intervention. 

This project presents a resource-efficient, interpretable Machine Learning (ML) framework for the automated classification and severity assessment of tomato leaf diseases. Unlike traditional "black-box" Deep Learning (DL) approaches that require massive datasets and high computational power, this solution utilises handcrafted feature extraction and classical ML models, making it highly suitable for edge deployment on mobile devices.

## ✨ Core Features & Application Workflow
The project is deployed via a user-friendly **Streamlit web interface** with the following capabilities:
* **Image Upload:** Users can upload images of tomato leaves for instant analysis.
* **Two-Stage Screening:** 1. *Binary Classification:* Identifies if the leaf is Healthy or Diseased.
  2. *Multiclass Classification:* categorises the specific infection into one of nine disease categories.
* **Severity Assessment:** Quantifies disease severity into Mild, Moderate, or Severe categories based on deviations from healthy baselines.
* **Treatment Chatbot:** Optional integration of an interactive chatbot to recommend specific agricultural interventions based on the diagnosed disease.

## 🧠 Methodology & Architecture

### 1. Feature Extraction
The pipeline extracts highly specific physical properties from resized images to generate a **133-dimensional feature vector**, capturing:
* **Color Properties**
* **Texture Characteristics**
* **Shape Dynamics**

### 2. Machine Learning Models
The framework was evaluated on the **PlantVillage dataset**. Several classifiers were trained and compared using ROC-AUC and F1-score metrics:
* Support Vector Machine (SVM) *(Best Performing Model)*
* Extreme Gradient Boosting (XGBoost)
* Random Forest (RF)
* k-Nearest Neighbours (kNN)
* Decision Trees (DT)

### 3. Model Interpretability
To ensure transparency and trust in the agricultural domain, model predictions are explained using **SHAP (SHapley Additive exPlanations) values**, mapping exactly which physical leaf features drove the model's final diagnosis.

## 💻 Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/tomato-leaf-app.git](https://github.com/your-username/tomato-leaf-app.git)
cd tomato-leaf-app
```

**2. Create a virtual environment**
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

**3. Install dependencies** 
```bash
pip install -r requirements.txt
```

**4. Run the web application** 
```bash
streamlit run app.py
```
### How to update your GitHub repo right now:
Since you already created the `README.md` file earlier, you can simply open it in your editor (or by typing `notepad README.md` in PowerShell), delete what is currently there, paste this new version in, and save it. 

Then, run your standard update cycle:
```powershell
git add README.md
git commit -m "Updated README with official project abstract and architecture"
git push origin main

