import os
from torchvision import transforms
from torch.utils.data import DataLoader
from training.custom_dataset import XrayMetadataDataset

def get_transforms(img_size=64):
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def create_dataloader(data_dir, batch_size=32, img_size=64, shuffle=True):
    """
    Create a standard data loader for the multi-modal dataset.
    """
    # Map class to metadata file (as requested)
    csv_files = {
        'COVID': os.path.join(data_dir, 'COVID.metadata.xlsx'),
        'Normal': os.path.join(data_dir, 'Normal.metadata.xlsx'),
        'Lung Opacity': os.path.join(data_dir, 'Lung_Opacity.metadata.xlsx'),
        'Viral Pneumonia': os.path.join(data_dir, 'Viral Pneumonia.metadata.xlsx')
    }
    
    transform = get_transforms(img_size)
    dataset = XrayMetadataDataset(data_dir=data_dir, csv_files=csv_files, transform=transform, img_size=img_size)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
