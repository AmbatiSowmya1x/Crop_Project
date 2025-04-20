# 🌾 Crop Recommendation and Yield Prediction System

This project provides a web-based interface for recommending suitable crops and predicting expected yields based on soil and weather inputs.

---

## 🚀 Features

- **Crop Recommendation**: Recommends suitable crops based on soil properties (Nitrogen, Potassium, Phosphorus, pH) and weather conditions (Temperature, Humidity, Rainfall).
- **Yield Prediction**: Predicts the crop yield based on soil and weather inputs.
- **Django-based Interface**: User-friendly web interface for inputting data and displaying results.
- **Machine Learning Models**: 
  - Crop Recommendation: Naïve Bayes, Random Forest, SVM, Logistic Regression, Decision Tree, XGBoost
  - Yield Prediction: Random Forest Regressor, Linear Regression, Decision Tree Regressor, SVR, KNN Regressor

---

## 🛠 Setup Instructions

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Crop_Recommendation_Project_repo

2. Create and activate a virtual environment
   ```bash
    conda create -n test_env python=3.10
    conda activate test_env

4. Install Dependencies
    ```bash
   pip install -r requirements.txt

6. Run migrations
   ```bash
    python manage.py makemigrations
    python manage.py migrate

8. Start the server
   ```bash
    python manage.py runserver

10. Visit http://127.0.0.1:8000/ in your web browser to access the system.


## Project Structure
```
Crop_Recommendation_Project_repo/
├── data/                  # Datasets for training and prediction
├── django_project/        # Django application code
│   ├── crop_app/          # Main app for crop recommendation and yield prediction
│   ├── django_project/    # Django project configurations
│   ├── db.sqlite3         # Database file
│   └── manage.py          # Django manage script
├── models/                # Folder for storing trained machine learning models
│   ├── crop_model.pkl     # Trained crop recommendation model
│   ├── yield_model.pkl    # Trained yield prediction model
│   └── scaler_encoder.pkl # Preprocessing scaler and encoder
├── notebooks/             # Jupyter notebooks for data exploration and model development
├── requirements.txt       # List of required Python packages
└── README.md              # Project documentation (this file)

