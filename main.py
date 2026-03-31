import os
import argparse
from training.train import train
from training.evaluate import evaluate

def main():
    
    parser = argparse.ArgumentParser(description="NeuroSyn using ViT")
    parser.add_argument('mode', type=str, choices=['train', 'evaluate'], help="Mode of execution: 'train' or 'evaluate'")
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'COVID-19_Radiography_Dataset'), help="Path to dataset directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training/eval")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--limit_batches', type=int, default=None, help="Limit number of batches for fast test")
    parser.add_argument('--model_path', type=str, default='model.pth', help="Path to save/load model")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Starting training on dataset at {args.data_dir}...")
        train(args.data_dir, batch_size=args.batch_size, epochs=args.epochs, limit_batches=args.limit_batches)
    elif args.mode == 'evaluate':
        print(f"Starting evaluation of model {args.model_path}...")
        evaluate(args.data_dir, model_path=args.model_path)

if __name__ == "__main__":
    main()
