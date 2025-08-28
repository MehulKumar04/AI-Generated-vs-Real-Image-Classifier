# 🎨 AI vs Real Image Classifier 🖼️

A deep learning project that classifies whether an image is **AI-generated** or a **Real Art/Painting/Photo**.  
This project uses **InceptionResNetV2** with custom layers and achieves **95% accuracy** on the dataset.  

It also includes a **Streamlit Web App** where users can upload an image and instantly see whether it is AI-generated or real.

---

## 🚀 Features
- Binary classification: **AI Art (1)** vs **Real Art (0)**
- Built with **TensorFlow / Keras**
- Achieved **95% accuracy** using **InceptionResNetV2**
- Easy-to-use **Streamlit app** for live predictions
- Supports **PNG, JPG, JPEG** images

---

## 📂 Dataset
The dataset consists of two folders:
- `AiArtData/` → AI-generated images
- `RealArt/` → Real artworks and photographs  

All images were resized to **224×224 pixels** and normalized between **0–1** for model training.  

📥 Dataset is available on Kaggle:  
👉 [AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images?select=AiArtData)  

---

## 🧠 Model Architecture
- **Base Model**: InceptionResNetV2 (pre-trained on ImageNet, frozen layers)  
- **Custom Layers**:
  - Global Average Pooling  
  - Dense(1024, ReLU)  
  - Dense(1, Sigmoid)  

Trained with **binary crossentropy loss** and **Adam optimizer**.  

---


