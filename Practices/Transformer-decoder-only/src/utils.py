import os
import math
import requests
import torch
from datasets import load_dataset
from huggingface_hub import login
import json

from model import Model

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)
login(config['huggingface-token']['access_token']) if config['huggingface-token']['access_token'] != "YOUR_HUGGING_FACE_ACCESS_TOKEN" else None

def load_data_with_huggingface(path: str):
    """
    Load training data using Hugging Face Datasets
    
    Args:
        path (str): Hugging Face dataset
    
    Returns:
        str: Loaded text data
    """
    dataset = load_dataset(path)
    return dataset

def load_data_with_url(url: str):
    """
    Load training data, downloading if not exists
    
    Args:
        url (str): Source URL for dataset
    
    Returns:
        str: Loaded text data
    """
    os.makedirs('./data', exist_ok=True)
    file_path = './data/sales_textbook.txt'
    
    if not os.path.exists(file_path):
        try:
            print(f"Downloading dataset from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print("Dataset downloaded successfully")
            
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

def prepare_data(data, train_split=0.8):
    """
    Prepare and split tokenized data
    
    Args:
        data (list): Tokenized data
        train_split (float): Proportion of data for training
    
    Returns:
        tuple: Training and validation data
    """
    num_train = math.ceil(train_split * len(data))
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    print(f"Data split: Train {len(train_data)} tokens, Validation {len(val_data)} tokens")
    return train_data, val_data

def get_batch(data, context_length, batch_size, device):
    """
    Generate training batches
    
    Args:
        data (list): Tokenized data
        config (TrainingConfig): Training configuration
    
    Returns:
        tuple: Input and target batches
    """
    idxs = torch.randint(
        low=0, 
        high=len(data) - context_length - 1, 
        size=(batch_size,)
    )
    x_batch = torch.stack([
        torch.tensor(data[idx:idx+context_length]) 
        for idx in idxs
    ]).to(device)
    
    y_batch = torch.stack([
        torch.tensor(data[idx+1:idx+context_length+1]) 
        for idx in idxs
    ]).to(device)
    
    return x_batch, y_batch

def calculate_parameter(model, path):
    model = Model()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    
    # calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
                

if __name__=='__main__':
    # raw_data = load_data_with_huggingface('goendalf666/sales-textbook_for_convincing_and_selling')
    # if raw_data is None:
    #     print("Failed to load training data")
    # print(f"Data loaded: {len(raw_data)} tokens")
    # print(raw_data['train']['text'])

    model = Model()
    total_params = calculate_parameter(model, './model/model.pth')
    print(f"Total parameters: {total_params}")