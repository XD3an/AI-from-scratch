import os
import logging
import torch
import torch.optim as optim
import json

from model import Model
from utils import load_data_with_huggingface, prepare_data, get_batch, calculate_parameter
from tokenizer import TextTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("./logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameters
class TrainingConfig:
    """Configuration for training the model"""
    def __init__(self, config_path: str = 'config.json'):
        """
        Load configuration from JSON file
        
        Args:
            config_path (str): Path to the configuration JSON file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.BATCH_SIZE = config['train']['batch_size']
        self.ITERATIONS = config['train']['iterations']
        self.LEARNING_RATE = config['train']['learning_rate']
        self.EVAL_INTERVAL = config['train']['eval_interval']
        self.CONTEXT_LENGTH = config['train']['context_length']
        self.SEED = config['train']['seed']
        self.DATASET_PATH = config['train']['dataset_path']
        self.MODEL_PATH = config['train']['model_path']
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

def train_model(model, optimizer, train_data, val_data, batch_size, iterations, eval_interval, context_length, device, seed):
    """
    Train the model
    
    Args:
        model (Model): Model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        train_data (list): Training data
        val_data (list): Validation data
        batch_size (int): Batch size
        iterations (int): Number of training iterations
        context_length (int): Context length for training
        device (str): Device to run the model on
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logger.info(f"Training on {device}")

    # Training loop
    for step in range(iterations):

        # evaluate model
        if step == 0 or step % eval_interval == 0:
            num_batches = eval_interval // batch_size  # number of batches to evaluate
            losses = estimate_loss(model, train_data, val_data, batch_size, num_batches, context_length, device)
            logger.info(
                f"[Iterations: {step:6d}] Train loss: {losses['train_loss']:8.4f} | Validation loss: {losses['val_loss']:8.4f}"
            )
            # checkpoint model
            torch.save(model.state_dict(), f"./model/model_{step}.pt")
        
        # get a batch of data
        x_batch, y_batch = get_batch(train_data, context_length, batch_size, device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass and compute loss
        _, loss = model(x_batch, y_batch)

        # backward pass and optimizer step
        loss.backward()
        optimizer.step()

def main():
    """Main training script"""
    try:
        config = TrainingConfig()
        
        # 1. Load and prepare data (parquet format)
        textbook = load_data_with_huggingface(path=config.DATASET_PATH)
        if textbook is None:
            logger.error("Failed to load training data")
            return
        
        try:
            train_textbook = []
            test_textbook = []
            for i in range(0, len(textbook['train'])):
                train_textbook.append(textbook['train'][i]['text'])
            train_textbook = ' '.join(train_textbook)
            
            for i in range(0, len(textbook['test'])):
                test_textbook.append(textbook['test'][i]['text'])
            test_textbook = ' '.join(test_textbook)
        except:
            pass
        
        if len(train_textbook) < config.CONTEXT_LENGTH or len(test_textbook) < config.CONTEXT_LENGTH:
            logger.error(f"Data length ({len(train_textbook)}/{len(test_textbook)}) is smaller than context length ({config.CONTEXT_LENGTH})")
            train_textbook, test_textbook = prepare_data(train_textbook)
        logger.info(f"Data loaded: {len(train_textbook)} training samples, {len(test_textbook)} test samples")
        
        # 2. Tokenize data
        tokenizer = TextTokenizer(encoding_name="cl100k_base")
        train_data = tokenizer.encode(train_textbook)
        val_data = tokenizer.encode(test_textbook)
        # train_data, val_data = prepare_data(tokenized_data)   # Split data into train and validation (80/20)
        logger.info(f"Data split: Train {len(train_data)} tokens, Validation {len(val_data)} tokens")
        
        # 3. Initialize model
        model = Model(
        ).to(config.DEVICE)
        logger.info("Model initialized")
        
        # 4. Optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE
        )
        logger.info("Optimizer initialized")
        
        # 5. Training loop
        train_model(model, 
                    optimizer, 
                    train_data, 
                    val_data, 
                    config.BATCH_SIZE, 
                    config.ITERATIONS, 
                    config.EVAL_INTERVAL, 
                    config.CONTEXT_LENGTH, 
                    config.DEVICE, 
                    config.SEED)
        logger.info("Training completed")
        
        # 6. Save the model
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), config.MODEL_PATH)
        logger.info("Model training completed and saved.")
        
        # Calculate and log the number of parameters
        total_params = calculate_parameter(model=model, path=config.MODEL_PATH)
        logger.info(f"Total parameters: {total_params}")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == '__main__':
    main()