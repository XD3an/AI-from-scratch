from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load tokenizer, model
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True).to(torch_device)   
    print(f"{model}, {tokenizer}")

    # inference
    input_text = "Hello, world!"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(torch_device)  # tokenizer encoding

    output = model.generate(input_ids, max_length=50, num_return_sequences=1) # get output

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True) # decode output

    print(generated_text)


if __name__ == "__main__":
    main()
