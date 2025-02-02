# AI from scratch

## Introduction

## Model

### Transformer-Decoder-only

實作 Transformer 的 Decoder-only 模型，用於根據給定 prompt 生成後續。

- TODO:
  - [x] combind [src/parameter.py]() to [src/utils.py]()
  - [x] change [src/generation/TextGenerator]() to [Inferencer]()
  - [x] change [src/evaluation]() method to properly
  - [] add [Fine-tuning]() method
    - finetune-dataset.json
      - [Hugging face 🤗 - Alpaca format Dataset reference](https://huggingface.co/datasets?sort=trending&search=Alpaca)
    - add fine-tuning method

#### Usage

- 可修改 `config.json` 中的參數，調整模型訓練的參數。

1. 安裝 requirements

    ```bash
    pip install -r requirements.txt
    ```

2. 訓練模型

    ```bash
    python src/train.py
    ```
    or 
    ```bash
    ./train.bat # for windows
    ```

3. 測試模型
    - 根據給定 prompt 生成後續
    
    ```bash
    python src/generation.py
    ```

## Reference

- [https://github.com/waylandzhang/Transformer-from-scratch](https://github.com/waylandzhang/Transformer-from-scratch)