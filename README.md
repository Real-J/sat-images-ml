# 🌍 Satellite Image Classification using Convolutional Neural Networks (CNN) 🚀

## 📌 Project Overview
This project uses **Deep Learning (CNNs)** to classify satellite images into four categories:
- 🌥 **Cloudy**
- 🏜 **Desert**
- 🌊 **Water**
- 🌿 **Green Areas**

We use **TensorFlow and Keras** to train a **Convolutional Neural Network (CNN)** on a dataset of satellite images, enabling the model to accurately identify different terrains and environmental conditions.

---

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/Real-J/satellite-image-classification.git
cd satellite-image-classification
```

### **2️⃣ Install Dependencies**
Ensure you have **Python 3.8+** installed. Then, install the required packages:
```sh
pip install tensorflow matplotlib numpy pandas opencv-python
```

### **3️⃣ Set Up Your Dataset**
Ensure your dataset is structured as follows:
```
dataset/
├── cloudy/
├── desert/
├── water/
├── green_area/
```
Modify the `DATA_PATH` in `satellite_cnn.py` to point to your dataset location:
```python
DATA_PATH = "/path/to/your/dataset"
```

---

## 🚀 Training the Model
Run the training script:
```sh
python satellite_cnn.py
```

### **Training Process**
The model trains for **30 epochs** using:
- **Data Augmentation** (flipping, rotation, zoom, contrast, brightness adjustments)
- **Batch Normalization & Dropout (0.6, 0.4)** to prevent overfitting
- **L2 Regularization** for stable weight updates
- **Early Stopping & Learning Rate Scheduler** for efficient convergence

---

## 🧠 Machine Learning Algorithm: Convolutional Neural Network (CNN)
### **1️⃣ What is a CNN?**
A **Convolutional Neural Network (CNN)** is a deep learning model designed to recognize patterns in images. It works by learning spatial hierarchies of features through:
1. **Convolutional Layers** → Extract feature maps from images.
2. **Pooling Layers** → Reduce dimensionality while preserving essential features.
3. **Fully Connected Layers** → Interpret extracted features for classification.

### **2️⃣ Our CNN Architecture**
The CNN model follows this structure:
1. **Input Layer**: Image size `(72x128x3)`
2. **3 Convolutional Blocks**
   - `Conv2D (32, 64, 128 filters)` with `ReLU activation`
   - `MaxPooling2D` for downsampling
   - `BatchNormalization` for faster training
3. **Flatten Layer**
4. **Fully Connected Layers**
   - `Dense(256 neurons)` with dropout (`0.6, 0.4`)
   - `Softmax Output (4 classes)`

### **3️⃣ Why Use CNNs for Satellite Images?**
- **Feature Extraction:** CNNs detect edges, textures, and patterns, making them ideal for satellite imagery.
- **Translation Invariance:** Pooling ensures robust learning regardless of the image’s position.
- **Scalability:** CNNs generalize well to large-scale datasets.

---

## 📊 Model Performance & Results
After **30 epochs**, the model achieves:
- **Training Accuracy:** `98.36%`
- **Validation Accuracy:** `98.67%`
- **Final Loss:** `0.0848`

### **Accuracy & Loss Graphs**
The model shows **minimal overfitting** and smooth convergence.

---

## 🛠 Next Steps & Deployment
### **1️⃣ Evaluate on a Test Dataset**
```python
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.4f}")
```

### **2️⃣ Make Predictions on New Images**
```python
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = load_model("best_model.keras")
img = tf.keras.preprocessing.image.load_img("test_image.jpg", target_size=(72, 128))
img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
predictions = model.predict(img_array)
print("Predicted Class:", CLASS_NAMES[np.argmax(predictions)])
```

### **3️⃣ Convert Model to TensorFlow Lite (For Edge Deployment)**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("satellite_model.tflite", "wb") as f:
    f.write(tflite_model)
```

### **4️⃣ Deploy as a Web API (Using FastAPI)**
```python
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("best_model.keras")
CLASS_NAMES = ["cloudy", "desert", "water", "green_area"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((72, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return {"prediction": CLASS_NAMES[np.argmax(prediction)]}

# Run API: `uvicorn app:app --reload`
```
----

#### Graphs:
![Graph](G_1.PNG)

![Graph](G_2.PNG)

---

## 📚 License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute!


