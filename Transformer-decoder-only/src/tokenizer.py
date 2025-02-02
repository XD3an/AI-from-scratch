import tiktoken
import json


with open("config.json", "r") as f:
    config = json.load(f)

class TextTokenizer:
    def __init__(self, encoding_name=config['tokenizer']['encoding_name']):
        """
        Initialize tokenizer with optional encoding selection
        
        Args:
            encoding_name (str): Tiktoken encoding name
        """
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            raise

    def encode(self, text):
        """
        Encode text to token ids
        
        Args:
            text (str): Input text to encode
        
        Returns:
            list: List of token ids
        """
        try:
            return self.tokenizer.encode(text)
        except Exception as e:
            print(f"Encoding error: {e}")
            return []

    def decode(self, token_ids):
        """
        Decode token ids back to text
        
        Args:
            token_ids (list): List of token ids
        
        Returns:
            str: Decoded text
        """
        try:
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            print(f"Decoding error: {e}")
            return ""

if __name__ == "__main__":
    tokenizer = TextTokenizer()
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)
    print(token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print(decoded_text)
