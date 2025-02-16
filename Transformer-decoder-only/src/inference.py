import torch
import logging
import tiktoken
import json

from model import Model
# from tokenizer import TextTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Hyperparameters
class InferenceConfig:
    """Configuration for Inference"""
    def __init__(self, config_path: str = 'config.json'):
        """
        Load configuration from JSON file
        
        Args:
            config_path (str): Path to the configuration JSON file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.MAX_TOKENS = config['inference']['max_tokens']
        self.TEMPERATURE = config['inference']['temperature']
        self.TOP_K = config['inference']['top_k']
InferenceConfig = InferenceConfig()

class Inferencer:

    def __init__(self, model_path, device=None):
        """
        Initialize text generator
        
        Args:
            model_path (str): Path to saved model weights
            device (str, optional): Computing device (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.model = Model().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
            logger.info("Model and tokenizer initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")

            raise
    
    def generate(self, 
                 ids=[],
                 max_tokens=InferenceConfig.MAX_TOKENS,
                 temperature=InferenceConfig.TEMPERATURE, 
                 top_k=InferenceConfig.TOP_K):
        """

        Generate text from a given prompt
        """
        try:
            # cover to tensor
            ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

            # Generate tokens
            y = self.model.generate(ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

            # Decode tokens to text
            generated_text = self.tokenizer.decode(y.squeeze().tolist())
            logger.info("Text generated successfully")
            return generated_text

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""

    def generate_text(self, 
                      prompt="", 
                      max_tokens=InferenceConfig.MAX_TOKENS,
                      temperature=InferenceConfig.TEMPERATURE, 
                      top_k=InferenceConfig.TOP_K):
        """
        Generate text from a given prompt
        

        Args:
            prompt (str): Starting text for generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k filtering for token selection
        
        Returns:
            str: Generated text
        """
        try:
            # Tokenize the prompt
            start_ids = self.tokenizer.encode(prompt)
        
            # cover to tensor
            start_ids = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(self.device)

            # Generate tokens
            y = self.model.generate(start_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

            # Decode tokens to text
            generated_text = self.tokenizer.decode(y.squeeze().tolist())
            logger.info("Text generated successfully")
            return generated_text

        
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""

def main():
    """Interactive text generation script with parameter controls"""
    try:
        generator = Inferencer(model_path='./model/model.pth')
        print("\nText Generation Interface")
        print("------------------------")
        
        while True:
            # Get prompt
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
                
            # Get generation parameters
            try:
                max_tokens = input("Enter max tokens (default 500): ").strip()
                max_tokens = int(max_tokens) if max_tokens else 500
                
                temperature = input("Enter temperature 0.0-1.0 (default 0.7): ").strip()
                temperature = float(temperature) if temperature else 0.7
                
                top_k = input("Enter top_k (default 50): ").strip()
                top_k = int(top_k) if top_k else 50
                
            except ValueError as e:
                print("Invalid input, using default values")
                max_tokens, temperature, top_k = 500, 0.7, 50
            
            print(f"\nGenerating text with parameters:")
            print(f"- Max tokens: {max_tokens}")
            print(f"- Temperature: {temperature}")
            print(f"- Top-k: {top_k}")
            print("\nGenerated text:")
            print("--------------")
            
            generated_text = generator.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            print(generated_text)
            print("--------------")
    
    except Exception as e:
        logger.error(f"Generation script failed: {e}")

if __name__ == '__main__':
    main()
