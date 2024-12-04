test_data = './data/val'
test_labels = './data/val.csv'

# Download utilities.py
import os
import requests

file_name = "utilities.py"
url = "https://raw.githubusercontent.com/mohiteamit/upGrad-Gesture-Recognition/refs/heads/main/utilities.py"

# Check if the file exists
if not os.path.exists(file_name):
    print(f"{file_name} not found. Downloading...")
    try:
        response = requests.get(url)
        if response.ok:
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"{file_name} downloaded successfully.")
        else:
            print(f"Failed to download {file_name}. HTTP Status Code: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        exit(1)

import os
import requests

# List of model URLs
model_urls = [
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/Conv2D+GRU.keras",
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/Conv2D+LSTM.keras",
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/Conv3D-32-64-128.keras",
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/pretrained-MobileNetV2+GRU.keras",
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/pretrained-MobileNetV3Small+GRU.keras",
    "https://github.com/mohiteamit/upGrad-Gesture-Recognition/raw/refs/heads/main/best-models/pretrained-mediapipe+gru.keras",
]

# Directory to save models
output_dir = "models_to_evaluate"
os.makedirs(output_dir, exist_ok=True)

# Function to verify file integrity
def verify_file(file_path, url):
    with open(file_path, 'rb') as f:
        local_content = f.read()
    response = requests.get(url)
    return response.ok and local_content == response.content

# Download models
for url in model_urls:
    filename = os.path.join(output_dir, os.path.basename(url))
    try:
        if not os.path.exists(filename) or not verify_file(filename, url):
            response = requests.get(url)
            if response.ok:
                with open(filename, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download: {url}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

print("Models downloaded.")

from utilities import GestureDataGenerator
import tensorflow as tf
from tensorflow.keras.models import load_model

if tf.__version__.startswith("2.10"):
    image_size = (120, 120)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size
    )

    Conv2D_GRU = load_model('models_to_evaluate/Conv2D+GRU.keras')                   # Best image size 120x120
    Conv2D_GRU.summary()
    evaluation_results = Conv2D_GRU.evaluate(test_generator)
    for metric, value in zip(Conv2D_GRU.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("Conv2D_GRU model requires TensorFlow 2.10.x")

if tf.__version__.startswith("2.10"):
    image_size = (120, 120)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size
    )

    Conv2D_LSTM = load_model('models_to_evaluate/Conv2D+LSTM.keras')
    Conv2D_LSTM.summary()
    evaluation_results = Conv2D_LSTM.evaluate(test_generator)
    for metric, value in zip(Conv2D_LSTM.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("Conv2D_LSTM model requires TensorFlow 2.10.x")

if tf.__version__.startswith("2.10"):
    image_size = (200, 200)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size
    )
    Conv3D_32_64_128 = load_model('models_to_evaluate/Conv3D-32-64-128.keras') 
    Conv3D_32_64_128.summary()
    evaluation_results = Conv3D_32_64_128.evaluate(test_generator)
    for metric, value in zip(Conv3D_32_64_128.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("Conv3D_32_64_128 model requires TensorFlow 2.10.x")

if tf.__version__.startswith("2.18"):
    image_size = (224, 224)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size,
        use_mediapipe=False
    )

    MobileNetV2_GRU = load_model('models_to_evaluate/pretrained-MobileNetV2+GRU.keras')
    MobileNetV2_GRU.summary()
    evaluation_results = MobileNetV2_GRU.evaluate(test_generator)
    for metric, value in zip(MobileNetV2_GRU.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("MobileNetV2_GRU model requires TensorFlow 2.18.x")

if tf.__version__.startswith("2.18"):
    image_size = (224, 224)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size,
    )

    MobileNetV3Small_GRU = load_model('models_to_evaluate/pretrained-MobileNetV3Small+GRU.keras')
    MobileNetV3Small_GRU.summary()
    evaluation_results = MobileNetV3Small_GRU.evaluate(test_generator)
    for metric, value in zip(MobileNetV3Small_GRU.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("MobileNetV3Small_GRU model requires TensorFlow 2.18.x")

if tf.__version__.startswith("2.18"):
    image_size = (256, 256)

    test_generator = GestureDataGenerator(
        data_path=test_data,
        labels_csv=test_labels,
        image_size=image_size,
        use_mediapipe=True
    )

    mediapipe_GRU = load_model('models_to_evaluate/pretrained-mediapipe+gru.keras')
    mediapipe_GRU.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    mediapipe_GRU.summary()
    evaluation_results = mediapipe_GRU.evaluate(test_generator)
    for metric, value in zip(mediapipe_GRU.metrics_names, evaluation_results):
        print(f"{metric}: {value:.4f}")
else:
    print("mediapipe_GRU model requires TensorFlow 2.18.x")
