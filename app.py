from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import joblib
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained SVM Model
svm_model = joblib.load("svm_model.pkl")

# Define CNN model architecture before loading weights
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 2)  # 2 classes (Original/Counterfeit)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize CNN model and load trained weights
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
cnn_model.eval()

# Function to preprocess images for SVM
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize
    img = img / 255.0  # Normalize
    return img.reshape(1, -1)  # Flatten for SVM

@app.route("/upload", methods=["POST"])
def upload_qr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image for SVM
    img_svm = preprocess_image(file_path)
    prediction_svm = svm_model.predict(img_svm)[0]  # SVM Prediction

    # Preprocess the image for CNN
    img_cnn = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_cnn = cv2.resize(img_cnn, (128, 128))
    img_cnn = torch.tensor(img_cnn, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(img_cnn)
        prediction_cnn = torch.argmax(output, dim=1).item()

    # Decide final result
    result = "Original QR Code" if prediction_cnn == 0 else "Counterfeit QR Code"

    return jsonify({"result": result, "svm_prediction": int(prediction_svm)})
@app.route("/")
def home():
    return "QR Code Authentication API is Running!"

if __name__ == "__main__":
    app.run(debug=True)
