from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open("E:/Crop_Recommendation_Project/models/best_crop_model.pkl", "rb") as f:
    crop_recommend_model = pickle.load(f)

with open('E:/Crop_Recommendation_Project/models/crop_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open("E:/Crop_Recommendation_Project/models/yield_prediction_model.pkl", "rb") as f:
    yield_model = pickle.load(f)

with open("E:/Crop_Recommendation_Project/models/yield_scaler.pkl", "rb") as f:
    x_scaler = pickle.load(f)

#crop_names = sorted(crop_encoder.inverse_transform(np.arange(len(crop_encoder.classes_))))

def index(request):
    return render(request, 'crop_app/index.html')

def crop_recommend(request):
    if request.method == "POST":
        # Get input values
        nitrogen = float(request.POST["nitrogen"])
        phosphorus = float(request.POST["phosphorus"])
        potassium = float(request.POST["potassium"])
        ph = float(request.POST["ph"])
        temperature = float(request.POST["temperature"])
        humidity = float(request.POST["humidity"])
        rainfall = float(request.POST["rainfall"])

        # Create input array
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Predict crop
        predicted_crop = crop_recommend_model.predict(input_data)[0]

        predicted_crop = label_encoder.inverse_transform([predicted_crop])[0]

        return render(request, "crop_app/crop_recommend.html", {"recommended_crop": predicted_crop})

    return render(request, "crop_app/crop_recommend.html")


def yield_predict(request):
    predicted_yield = None

    if request.method == "POST":
        # Collect input values from the form
        rainfall = float(request.POST["rainfall"])
        fertilizer = float(request.POST["fertilizer"])
        temperature = float(request.POST["temperature"])
        nitrogen = float(request.POST["nitrogen"])
        phosphorus = float(request.POST["phosphorus"])
        potassium = float(request.POST["potassium"])

        # Arrange input in the correct order
        input_data = np.array([[rainfall, fertilizer, temperature, nitrogen, phosphorus, potassium]])

        # Scale the input
        input_scaled = x_scaler.transform(input_data)

        # Predict the yield
        predicted_yield = yield_model.predict(input_scaled)[0]
        predicted_yield = round(predicted_yield, 2)

        return render(request, "crop_app/yield_predict.html", {"predicted_yield": predicted_yield})

    return render(request, "crop_app/yield_predict.html")

