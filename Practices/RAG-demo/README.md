# RAG demo

RAG 模型 demo (中文，基於 BGE 模型，醫療領域)。

## Usage

1. build the virtual environment

    ```
    python -m venv .venv

    # activate
    .venv/Scripts/activate     # Windows
    source .venv/bin/activate  # macOS and Linux
    
    # install required packages

    pip install -r requirements.txt

    ```

2. Run the demo

    ```bash
    python3 main.py
    ```

## Embedding Model

- [⚡️BGE: One-Stop Retrieval Toolkit For Search and RAG](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)
    - [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)

## References

- [wyf3/llm_related/rag_demo](https://github.com/wyf3/llm_related/tree/main/rag_demo)