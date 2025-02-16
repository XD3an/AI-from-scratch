import torch
import logging
import json
import os

from model import Model
from utils import load_data_with_huggingface, prepare_data, get_batch, calculate_parameter
from tokenizer import TextTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Hyperparameters
class FinetuneConfig:
    """Configuration for finetuning the model"""
    
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
            
        self.DATASET_PATH = config['finetune']['dataset_path']
        self.MODEL_PATH = config['finetune']['model_path']
        self.TARGET_MODEL_PATH = config['finetune']['target_model_path']
        self.BATCH_SIZE = config['finetune']['batch_size']
        self.ITERATIONS = config['finetune']['iterations']
        self.LEARNING_RATE = config['finetune']['learning_rate']
        self.EVAL_INTERVAL = config['finetune']['eval_interval']
        self.CONTEXT_LENGTH = config['finetune']['context_length']
        self.SEED = config['finetune']['seed']
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FinetuneConfig = FinetuneConfig()

def format_data(example):
    prompt = f"### Instruction:\n{example['instruction']}"
    if example['input']:
        prompt += f"### Input:\n{example['input']}"
    prompt += f"### Output:\n{example['output']}"
    return prompt

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, num_batches, context_length, device):
    """
    Estimate model loss on training and validation sets
    
    Args:
        model (Model): Trained model
        train_data (list): Training data
        val_data (list): Validation data
        batch_size (int): Batch size
        eval_interval (int): Interval for evaluation
        context_length (int): Context length for training
        device (str): Device to run the model on
    
    Returns:
        dict: Losses for training and validation sets
    """
    out = {}
    model.eval()

    # Estimate loss on training set
    for split in ['train', 'val']:
        losses = torch.zeros(num_batches)
        for k in range(num_batches):
            data = train_data if split == 'train' else val_data
            x_batch, y_batch = get_batch(data, context_length, batch_size, device)
            _, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[f"{split}_loss"] = losses.mean().item()

    model.train()
    return out

def finetune(model, train_data, val_data, batch_size, context_length, learning_rate, iterations, eval_interval, device, seed):
    """
    Finetune the model
    
    Args:
        model (Model): Trained model
        train_data (list): Training data
        val_data (list): Validation data
        batch_size (int): Batch size
        context_length (int): Context length
        learning_rate (float): Learning rate
        iterations (int): Number of iterations
        eval_interval (int): Evaluation interval
        device (str): Device to run the model on
        seed (int): Seed for reproducibility

    Returns:
        Model: Finetuned model
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # AdamW optimizer (better than Adam)
    
    for iter in range(iterations):
        if iter % eval_interval == 0:
            num_batches = eval_interval // batch_size
            losses = estimate_loss(model, train_data, val_data, batch_size, num_batches, context_length, device)
            logger.info(f"[Iteration {iter: 6d}] train loss: {losses['train_loss']:.4f}, val loss: {losses['val_loss']:.4f}")

        x_batch, y_batch = get_batch(train_data, context_length, batch_size, device)
        logits, loss = model(x_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
            
            
def main():
    # Load data (alpaca format)
    try:
        finetune_datasets = load_data_with_huggingface(FinetuneConfig.DATASET_PATH)
        raw_train_daat = finetune_datasets["train"]
        formatted_train_data = [format_data(item) for item in raw_train_daat]
    except:
        logger.error("Failed to load training data")
        return

    # preprocess
    train_data, val_data = prepare_data(formatted_train_data)
    train_data = str(train_data)
    val_data = str(val_data)

    #print(train_data)
    #input()

    # Tokenize data
    tokenizer = TextTokenizer(encoding_name="cl100k_base")
    train_data = tokenizer.encode(train_data)
    val_data = tokenizer.encode(val_data)

    # Finetune model
    model = Model().to(FinetuneConfig.DEVICE)
    model.load_state_dict(torch.load(FinetuneConfig.MODEL_PATH))
    logger.info(f"Finetuning model: {FinetuneConfig.MODEL_PATH}")
    logger.info(f"Finetuning on {FinetuneConfig.DEVICE}")
    logger.info(f"Start finetuning...")
    finetuned_model = finetune(model, 
                               train_data, val_data, 
                               FinetuneConfig.BATCH_SIZE,
                               FinetuneConfig.CONTEXT_LENGTH, 
                               FinetuneConfig.LEARNING_RATE, 
                               FinetuneConfig.ITERATIONS, 
                               FinetuneConfig.EVAL_INTERVAL, 
                               FinetuneConfig.DEVICE, 
                               FinetuneConfig.SEED)
    logger.info(f"Finetuning completed.")

    # Save finetuned model
    os.makedirs('./model', exist_ok=True)
    torch.save(finetuned_model.state_dict(), FinetuneConfig.TARGET_MODEL_PATH)


if __name__ == "__main__":
    main()
