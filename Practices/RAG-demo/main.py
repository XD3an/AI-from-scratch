from dataclasses import dataclass
from typing import List
import jieba
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
import torch
from huggingface_hub import snapshot_download
import os


@dataclass
class RAGConfig:
    # Data configuration
    data_path: str = "./RAG_data/"
    file_pattern: str = "**/*.txt"
    encoding: str = "utf-8"
    
    # Text splitting configuration
    chunk_size: int = 500
    chunk_overlap: int = 0
    
    # Embedding configuration
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_model_path: str = "./models/bge-large-zh-v1.5"
    
    # Retrieval configuration
    top_k: int = 10
    rrf_k: int = 60
    
    # LLM configuration
    llm_model: str = "deepseek-r1:1.5b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    
    # prompt
    # prompt = '''
        # Task: Answer user questions based on retrieved documents
        # Requirements:
        #     1. Do not deviate from the retrieved document to answer the question
        #     2. If the retrieved document does not contain the answer to the user's question, please answer "I don't know"
        
        # User question:
        # {}
        
        # Retrieved documents:
        # {}
        # '''
    prompt = '''
    任務目標：根據檢索出的文檔回答用戶問題
    任務要求：
        1、不得脫離檢索出的文檔回答問題
        2、若檢索出的文檔不包含用戶問題的答案，請回答我不知道

    用戶問題：
    {}

    檢索出的文檔：
    {}
    '''

def download_model_if_not_exists(model_name="BAAI/bge-large-zh-v1.5", local_dir="./models/bge-large-zh-v1.5"):
    if not os.path.exists(local_dir):
        print(f"Downloading model {model_name} to {local_dir}...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            ignore_patterns=["*.md", "*.txt"]
        )
    return local_dir

def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))

class RAGSystem:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.vectorizer = None
        self.db = None
        self.llm = None
        self.texts = None
    
    def setup(self):
        # 1. load documents
        documents = self._load_documents()
        
        # 2. split documents
        docs = self._split_documents(documents)
        
        # 3. setup vector retriever
        self.vectorizer = self._setup_vector_retriever(docs)
        
        # 3. setup vector db
        self.db = self._setup_vector_db(docs)
        
        # 4. setup LLM
        self.llm = self._setup_llm()
        
        self.texts = [i.page_content for i in docs]

    def _load_documents(self):
        # load all documents from directory path
        try:
            print("[*] Loading documents...")
            if os.path.isfile(self.config.data_path):
                # load single file
                loader = TextLoader(self.config.data_path, encoding=self.config.encoding)
                documents = loader.load()
            else:
                # load all files in directory
                loader = DirectoryLoader(
                    self.config.data_path,
                    glob=self.config.file_pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': self.config.encoding},
                    show_progress=True,
                    use_multithreading=True
                )
                documents = loader.load()
                print(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            raise Exception(f"Error loading documents: {str(e)}")

    def _split_documents(self, documents):
        print("[*] Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=['\n']
        )
        docs = text_splitter.split_documents(documents)
        return docs

    def _setup_vector_retriever(self, docs):
        print("[*] Setting up vector retriever...")
        texts = [i.page_content for i in docs]
        texts_processed = [preprocessing_func(t) for t in texts]
        vectorizer = BM25Okapi(texts_processed)
        return vectorizer

    def _setup_vector_db(self, docs):
        print("[*] Setting up vector retriever...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model_path = download_model_if_not_exists(self.config.embedding_model_name, self.config.embedding_model_path)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                'device': device,
                'trust_remote_code': True
            }
        )
        
        return FAISS.from_documents(docs, embeddings)

    def _setup_llm(self):
        print("[*] Setting up LLM...")
        return ChatOpenAI(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key
        )
    
    def _rrf(self, vector_results: List[str], text_results: List[str], k: int=10, m: int=60):
        doc_scores = {}
        for rank, doc_id in enumerate(vector_results):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
        for rank, doc_id in enumerate(text_results):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
        sorted_results = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        return sorted_results

    def query(self, query: str, k: int=10):
        # 1. BM25 retrieval
        bm25_res = self.vectorizer.get_top_n(
            preprocessing_func(query),
            self.texts,
            n=k
        )
        
        # 2. Vector retrieval
        vector_res = self.db.similarity_search(query, k=k)
        vector_results = [i.page_content for i in vector_res]
        
        # 3. Re-rank results
        rrf_res = self._rrf(vector_results, bm25_res, k=k, m=self.config.rrf_k)
        
        # 4. Generate answer
        prompt = self.config.prompt.format(query, ''.join(rrf_res))
        response = self.llm.invoke(prompt)
        return response.content

def main():
    try:
            # iniiialize RAG system
            print("[*] 正在初始化RAG系統...")
            config = RAGConfig()
            rag = RAGSystem(config)
            rag.setup()
            print("[*] RAG系統初始化完成")
            
            # process query
            query = "失眠了應該怎麼辦"
            print("\n[*] 處理查詢:", query)
            answer = rag.query(query, k=2)
            print("\n[*] 回答:\n", answer)

    except Exception as e:
        print(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()