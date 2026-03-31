import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class XrayMetadataDataset(Dataset):
    """
    Custom PyTorch Dataset for loading X-ray images and patient metadata.
    Maps image file name to metadata using CSV files.
    """
    def __init__(self, data_dir, csv_files, transform=None, img_size=64):
        """
        Args:
            data_dir: Base directory for the dataset.
            csv_files: Dictionary mapping class name to its corresponding CSV file path.
            transform: Optional transform to be applied on an image.
            img_size: Size for image resizing.
        """
        self.data_dir = data_dir
        self.csv_files = csv_files
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Define disease classes and index mapping
        self.classes = ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all metadata
        self.metadata_dfs = {}
        for cls, csv_path in csv_files.items():
            if os.path.exists(csv_path):
                # We assume metadata files are either CSV or XLSX based on directory content
                if csv_path.endswith('.xlsx'):
                    df = pd.read_excel(csv_path)
                else:
                    df = pd.read_csv(csv_path)
                
                # Normalize column naming (some datasets use 'FILE NAME' or 'filename')
                df.columns = [c.upper() for c in df.columns]
                self.metadata_dfs[cls] = df
            else:
                self.metadata_dfs[cls] = pd.DataFrame()
        
        # Build image list
        self.image_samples = []
        for cls in self.classes:
            idx = self.class_to_idx[cls]
            cls_dir = os.path.join(data_dir, cls, 'images')
            if not os.path.exists(cls_dir):
                # Try without 'images' subfolder if it's missing
                cls_dir = os.path.join(data_dir, cls)
                
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_samples.append({
                            'path': os.path.join(cls_dir, img_name),
                            'label': idx,
                            'class': cls,
                            'filename': img_name
                        })

    def __len__(self):
        return len(self.image_samples)

    def __getitem__(self, i):
        sample = self.image_samples[i]
        
        # Load image
        img = Image.open(sample['path']).convert('L') # Convert to grayscale
        if self.transform:
            img = self.transform(img)
            
        # Extract metadata
        cls = sample['class']
        filename = sample['filename']
        # Remove extension for matching if needed (some CSVs have it, some don't)
        base_filename = os.path.splitext(filename)[0]
        
        df = self.metadata_dfs[cls]
        
        # Default metadata values (Handle missing values gracefully)
        age = 0.5  # Normalized default
        gender = 0.0 # 0 for male, 1 for female (example encoding)
        fever = 0.0
        cough = 0.0
        sob = 0.0
        
        # Attempt to find metadata row (matching on filename)
        if not df.empty and 'FILE NAME' in df.columns:
            # Match by case-insensitive name
            match = df[df['FILE NAME'].astype(str).str.contains(base_filename, case=False, na=False)]
            if not match.empty:
                # If specific columns exist, use them:
                if 'AGE' in df.columns:
                    age = float(match.iloc[0]['AGE']) / 100.0 # simple normalization
                if 'GENDER' in df.columns:
                    gender = 1.0 if str(match.iloc[0]['GENDER']).lower() == 'female' else 0.0
                
                # New Symptom Columns (if they exist in dataset)
                if 'FEVER' in df.columns:
                    fever = 1.0 if str(match.iloc[0]['FEVER']).lower() in ['yes', '1', '1.0', 'true'] else 0.0
                if 'COUGH' in df.columns:
                    cough = 1.0 if str(match.iloc[0]['COUGH']).lower() in ['yes', '1', '1.0', 'true'] else 0.0
                if 'SOB' in df.columns:
                    sob = 1.0 if str(match.iloc[0]['SOB']).lower() in ['yes', '1', '1.0', 'true'] else 0.0

        # Metadata Tensor: [Age, Gender, Fever, Cough, SOB]
        metadata_tensor = torch.tensor([age, gender, fever, cough, sob], dtype=torch.float32)
        
        return img, metadata_tensor, sample['label']
