# NeuroSyn

This project implements a complete end-to-end deep learning system that combines X-ray image analysis with patient metadata (age, gender) to predict disease classes.

## 🎯 Features
- **Manual ViT Implementation**: A Vision Transformer built from scratch using PyTorch `nn.TransformerEncoder`.
- **Multi-Modal Fusion**: Combines image features (Vision Transformer) with patient metadata (MLP) for enhanced diagnostic accuracy.
- **Disease Classes**: COVID, Normal, Lung Opacity, Viral Pneumonia.
- **Streamlit UI**: An interactive web application for image upload and real-time diagnosis.

## 📁 Project Structure
- `models/`: Neural network architectures (ViT, Metadata Encoder, Fusion Model).
- `training/`: Dataset loading and training/evaluation logic.
- `utils/`: Preprocessing and data pipeline utilities.
- `app/`: Streamlit web application.
- `main.py`: CLI entry point for training and evaluation.

## 🚀 Getting Started

### 1. Install Dependencies for project
```bash
pip install -r requirements.txt
```

### 2. Train the Model
To start training on the radiography dataset:
```bash
python main.py train --epochs 5 --batch_size 8
```
*Note: Use `--limit_batches 10` for a quick test run.*

### 3. Evaluate the Model
To check performance on the test set:
```bash
python main.py evaluate
```

### 4. Train X-ray Validation Model (Binary)
To train a dedicated validator for **X-ray vs non-X-ray** uploads:
```bash
python training/train_xray_validator.py --data_dir data/xray_validation --epochs 8 --batch_size 32
```
Expected dataset structure:
```text
data/xray_validation/
  non_xray/
    img1.jpg
    img2.png
  xray/
    scan1.png
    scan2.jpg
```
This creates `xray_classifier.h5` in the project root, which is used by `utils/validator.py`.

### 5. Launch the Web UI
Run the Streamlit app to interact with the model:
```bash
streamlit run app/app.py
```

## 🛠️ Requirements
- Python 3.8+
- PyTorch
- Streamlit
- Pandas & openpyxl 
- Pillow

## 📝 Dataset
The system expects the **COVID-19 Radiography Dataset** structure in `data/COVID-19_Radiography_Dataset/`. It automatically maps image filenames to the corresponding metadata in the provided XLSX/CSV files.
