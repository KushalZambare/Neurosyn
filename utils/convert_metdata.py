import os
import pandas as pd

def convert_dataset_xlsx_to_csv(data_dir):
    
    files = [
        'COVID.metadata.xlsx',
        'Normal.metadata.xlsx',
        'Lung_Opacity.metadata.xlsx',
        'Viral Pneumonia.metadata.xlsx'
    ]
    
    for f in files:
        xlsx_path = os.path.join(data_dir, f)
        if os.path.exists(xlsx_path):
            csv_path = xlsx_path.replace('.xlsx', '.csv')
            print(f"Converting {f} to CSV...")
            try:
                df = pd.read_excel(xlsx_path)
                df.to_csv(csv_path, index=False)
                print(f"✅ Created: {csv_path}")
            except Exception as e:
                print(f"❌ Error converting {f}: {e}")
        else:
            print(f"⚠️ Metadata file {f} not found in {data_dir}")

if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'COVID-19_Radiography_Dataset')
    convert_dataset_xlsx_to_csv(DATA_PATH)
