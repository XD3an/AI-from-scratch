from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import loda_dataset


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load tokenizer, model
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True).to(torch_device)   
    print(f"{model}, {tokenizer}")

    # load datasets
    

    # finetune
    


if __name__ == "__main__":
    main()
