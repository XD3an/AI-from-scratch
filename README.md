# AI from scratch

## Tranditional Machine Learning Fundamental

- **Regression methods**

  - Linear Regression（線性迴歸）：預測連續值的最基本方法，尋找數據之間的線性關係。
  - Polynomial Regression（多項式迴歸）：當數據關係非線性時使用，透過添加次方項來擬合曲線。
  - Ridge/Lasso Regression（嶺迴歸/套索迴歸）：處理多重共線性問題的正則化線性迴歸方法，透過懲罰項來減少過擬合。
  - Support Vector Regression (SVR)（支持向量迴歸）：利用支持向量找到最大容忍誤差的函數，可處理非線性關係。
  - Decision Tree Regression（決策樹迴歸）：基於樹狀結構進行預測，將數據分割成不同區域。
  - Random Forest Regression（隨機森林迴歸）：結合多個決策樹的集成學習方法，提高穩定性和準確性。

- **Classifications & Clustering methods**

  - Logistic Regression（邏輯迴歸）：用於二分類問題的統計方法，預測事件發生的概率。
  - Support Vector Machines (SVM)（支持向量機）：在特徵空間中找尋最佳分隔超平面來分類數據。
  - Decision Trees（決策樹）：樹狀結構模型，根據特徵值進行分支決策。
  - Random Forests（隨機森林）：多個決策樹的集成，減少過擬合風險。
  - Naive Bayes（樸素貝葉斯）：基於貝葉斯定理的分類器，假設特徵之間相互獨立。
  - K-Nearest Neighbors (KNN)（K近鄰）：基於相似性的分類方法，根據最近的K個鄰居進行分類。
  - K-Means Clustering（K均值聚類）：將數據分成K個不同的群組，最小化群內點到中心的距離和。
  - Hierarchical Clustering（層次聚類）：創建數據點的嵌套聚類，可以是自下而上或自上而下的方法。
  - DBSCAN（基於密度的聚類方法）：基於密度的聚類演算法，能發現任意形狀的聚類，對噪聲數據處理良好。
  - Gradient Boosting Machines (GBM)（梯度提升機）：一種集成方法，通過構建弱學習器序列來提高性能。
  - XGBoost, LightGBM, CatBoost：梯度提升的高效實現，針對不同場景優化的算法。

- **Dimensionality Reduction methods**

  - Principal Component Analysis (PCA)（主成分分析）：通過找出最大方差方向將高維數據投影到低維空間。
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)（t分布隨機鄰域嵌入）：非線性降維技術，特別適合可視化高維數據。
  - Linear Discriminant Analysis (LDA)（線性判別分析）：監督式降維技術，尋找能最佳區分不同類別的投影方向。
  - Singular Value Decomposition (SVD)（奇異值分解）：矩陣分解技術，用於壓縮和降噪。
  - Non-negative Matrix Factorization (NMF)（非負矩陣分解）：將矩陣分解為非負因子，適用於特徵提取。
  - Uniform Manifold Approximation and Projection (UMAP)（均勻流形近似與投影）：現代非線性降維技術，保留全局和局部結構。
  - Autoencoders（自編碼器）：神經網路架構，通過學習數據的壓縮表示實現降維。

## Deep Learning Fundamental

  - Multilayer Perceptron (MLP)（多層感知器）：最基本的前饋神經網路，由輸入層、一個或多個隱藏層和輸出層組成。每個神經元與下一層的所有神經元相連，用於分類和回歸任務。
  - Convolutional Neural Network (CNN)（卷積神經網路）：專為處理網格狀數據（如圖像）設計的神經網路。使用卷積操作自動提取特徵，通過卷積層、池化層和全連接層構成，在計算機視覺領域表現優異。
  - Recurrent Neural Network (RNN)（循環神經網路）：設計用於處理序列數據的神經網路。網絡中的神經元可以記住之前的訊息，適合處理時間序列、文本等序列數據。
  - Long Short-Term Memory (LSTM)（長短期記憶網路）：RNN的一種變體，解決了普通RNN的長期依賴問題。通過引入門控機制（輸入門、遺忘門、輸出門）來控制訊息流，有效處理長序列數據。
  - Variational Autoencoder (VAE)（變分自編碼器）：生成模型的一種，結合了自編碼器和機率模型。透過學習數據的隱變量表示，可以生成新的數據樣本，廣泛應用於圖像生成和特徵學習。
  - Generative Adversarial Networks (GAN)（生成對抗網路）：由生成器和判別器組成的對抗性架構。生成器嘗試創建逼真的數據，判別器嘗試區分真實和生成的數據，二者相互競爭提升。應用於圖像生成、風格轉換等領域。
  - Transformer：一種基於自注意力機制的神經網路架構，最初為自然語言處理設計。不依賴循環結構，通過注意力機制直接建模長距離依賴關係，展現出卓越的並行處理能力。是現代大語言模型（如GPT、BERT）的基礎架構。
    - [Transformer-Decoder-only](Practice/Transformer-decoder-only/README.md)

## AI Framework and Ecosystem

### Deep Learning Frameworks

- [TensorFlow](https://www.tensorflow.org/guide)：TensorFlow 是一個開源的機器學習框架，由 Google Brain 團隊開發。提供了一個用於構建和訓練深度神經網路的 API，並且支持多種硬體加速器，包括 GPU 和 TPU。TensorFlow 提供了一個靈活的計算圖模型，可以用於構建各種不同類型的機器學習模型，包括卷積神經網路、循環神經網路、自編碼器、生成對抗網路等。

- [PyTorch](https://pytorch.org/get-started/locally/)：PyTorch 是一個開源的深度學習框架，由 Facebook 的 AI 研究團隊開發。提供了一個用於構建和訓練深度神經網路的 API，並且支持 GPU 和 TPU 加速。

- [Keras](https://keras.io/)：Keras 是一個深度學習框架，可以運行在 TensorFlow、Theano 和 CNTK 等後端。Keras 提供了一個簡單而直觀的 API，可以用於構建和訓練深度神經網路。Keras 的設計目標是讓用戶能夠快速地構建和訓練深度學習模型，而不需要深入了解底層的實現細節。

### Fine-tuning

- [unsloth](https://github.com/unslothai/unsloth)：一個用於微調深度學習模型的工具，可以用於微調預訓練模型，並且支持多種不同類型的前沿模型。

### Model Compression

- [Quantization]()：量化是透過將模型權重和運算從高精度（例如 FP32）轉換為低精度（例如 INT8、BF16）來減少計算資源需求。

- [Pruning]()：剪枝技術移除影響較小的權重或神經元，以減少模型規模並加快運算速度。

- [Knowledge Distillation]()：知識蒸餾透過讓小模型（學生模型）學習大模型（教師模型）的輸出來獲得相似的性能。

- [Binarization]()：將模型權重和激活值約束為 0 或 1，以極大減少計算需求（例如 XNOR-Net）。

### Inference and Deployment

#### Inference

- [WebLLM](https://webllm.mlc.ai/)：是一個用於在瀏覽器中運行大型語言模型（LLMs）的推理引擎工具，可以用於生成文本、回答問題等。

- [LMStudio](https://lmstudio.ai/)：是一個功能齊全的本地部屬 LLM 運行環境框架，支援本地設備上離線執行大型語言模型。

- [Ollama](https://ollama.com/)：是一個開源的 LLM 服務框架，用於本地部署，且提供完整的模型管理與推理服務，適用於對資料安全性要求較高的應用環境。

- [vLLM](https://docs.vllm.ai/en/latest/index.html)：vLLM 是一個開源的高效能推理框架，專門針對大規模語言模型（LLMs）在 GPU 上的推理進行優化。其目的是提高大模型推理的速度和效率，特別是在處理大規模神經網路模型時。

- [LightLLM](https://github.com/ModelTC/lightllm)：是一個基於 Python 的輕量級大型語言模型 (LLM) 推理和服務框架。它設計上註重高效能與低資源消耗，適用於在資源受限的環境中（如行動裝置或邊緣運算）快速部署 LLM。

- [OpenLLM](https://github.com/bentoml/OpenLLM)：允許開發人員使用單一命令運行任何開源 LLM（Llama 3.3、Qwen2.5、Phi3 等）或自訂模型作為與 OpenAI 相容的 API。

- [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/en/index)：是一個用於部署和服務大型語言模型（LLM）的工具包。
    - [HuggingFace-Transformers/gpt2-inference/](https://www.notion.so/AI%20Framework%20and%20Ecosystem/HuggingFace-Transformers/)

- [GPT4ALL](https://www.nomic.ai/gpt4all)：是一個旨在讓大型語言模型（LLM）在本地設備上運行的開源平台。

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：是一個開源的 C/C++ 函式庫，旨在在各種硬體上有效率地進行大型語言模型（LLM）的推理。

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)：是 NVIDIA 開源的 TensorRT 工具箱，專門用於優化大型語言模型（LLM）的推理效能。它提供 Python API 和 C++ 元件，允許使用者有效地在 NVIDIA GPU 上建置和執行 TensorRT 推理引擎。

#### Interactions

##### Prompt

- [Prompting Guide](https://www.promptingguide.ai/)

- [Promptify](https://app.promptify.com/)

- [AIDungeon](https://aidungeon.com/)

- [Promptbase](https://promptbase.com/)

##### Workflow

- [n8n](https://github.com/n8n-io/n8n)：是一個開源的工作流自動化工具，可以用於構建和執行機器學習工作流。

- [dify](https://github.com/langgenius/dify)：是一個用於構建和部署大型語言模型（LLM）的工作流自動化工具，支持多種不同類型的模型和任務。

##### Agents

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)：MCP 是一個用於定義和交換模型上下文的協議，用於描述模型的環境、任務和目標，以及模型的訓練和推理過程。
  - [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
  - [MCP servers githuModel Context Protocol servers](https://github.com/modelcontextprotocol/servers)
  - [PulseMCP](https://www.pulsemcp.com/)
  - [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
  - MCP
    - [Filesystem MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
    - [Fetch MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
    - [Github MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github)
    - [mcp-installer](https://github.com/anaisbetts/mcp-installer)
    - [Claude Desktop Commander MCP](https://github.com/wonderwhy-er/ClaudeComputerCommander)
    - [mcp-ollama](https://github.com/emgeee/mcp-ollama)
    - ...

- [Github Copilot](https://github.com/features/copilot)：GitHub Copilot 是一個 AI 程式碼助手，可以幫助開發人員快速生成程式碼。

- [Cline](https://www.cline.ai/)：Cline 提供了一個靈活且可擴展的框架，支持多種 AI 模型和 MCP 功能，幫助開發者快速構建和部署智慧代理系統。
  - [Roo-Code](https://github.com/RooVetGit/Roo-Code)

- [Deep Research]()：是一種利用 AI agent 來自動執行多步驟網路研究的功能。讓 AI 從網路上自動抓取並分析大量資料（例如文本、圖像、PDF等），然後進行邏輯推理與綜合，最終生成一份類似於專業研究分析師所撰寫的詳盡報告。
  - [node-DeepResearch](https://github.com/jina-ai/node-DeepResearch)

- [browser-use](https://github.com/browser-use/browser-use)：一個用於使 AI 代理可以訪問網頁的工具。

- [crawl4ai](https://github.com/unclecode/crawl4ai)：用於幫助 AI 爬取相關網站的工具。

### Tools and Repositories

#### AI Tools

- 基礎大語言模型
  
  - [ChatGPT](https://chat.openai.com/)
  
  - [Claude](https://claude.ai/)

  - [Gemini](https://gemini.google.com/app)

  - [xAI](https://x.ai/)

  - [Meta Llama Llama](https://www.llama.com/)

  - [DeepSeek](https://www.deepseek.com/)

- AI 搜尋

  - [Perplexity AI](https://www.perplexity.ai/)

- AI 畫圖

  - [midjourney](https://www.midjourney.com/)

  - [Stability AI](https://stability.ai/) 

  - [Flux AI](https://flux1.ai/)

  - [whisk](http://labs.google/whisk) 

  - [Recraft](https://www.recraft.ai/)

- AI 影片

  - [Sora](https://openai.com/sora/)

  - [Runway](https://runwayml.com/research/gen-2)

  - [Pika](https://pika.art/)

- AI 語音

  - [ElevenLabs](https://elevenlabs.io/)

  - [ChatTTS](https://chattts.com/)

- AI 音樂

  - [Suno](https://suno.com/)

- AI PPT

  - [Gamma](https://gamma.app/)

- AI Coding

  - [Cursor](https://www.cursor.com/)

  - [Windsurf](https://codeium.com/windsurf)

- 其他實用工具

  - [Monica](https://monica.im/)

  - [NotebookLm](https://notebooklm.google.com/)

  - [沉浸式翻譯](https://immersivetranslate.com/)

  - [HeyGen](https://www.heygen.com/)


#### Collections

- [awesome-ai](https://github.com/openbestof/awesome-ai)：收集各種 AI 相關的資源和工具。

- [ai-collection](https://github.com/ai-collection/ai-collection)：一個 AI 工具和專案的集合。

- [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)：收集各種大型語言模型（LLM）的推理工具和框架。

- [Awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resourses)：收集各種大型語言模型（LLM）的相關資源。

- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)：Model Context Protocol (MCP) servers awesome list.

- [Awesome-AI-Agents](https://github.com/Jenqyang/Awesome-AI-Agents)：收集各種 AI-Agents 相關資訊。

## Practices

- [MNIST]()：MNIST 是一個手寫數字圖像數據集，包含 0 到 9 的 70,000 張 28x28 像素的灰度圖像。
  - [Practice MNIST/](Practices/MNIST/)

- [MNIST-distillation]()：MNIST-distillation 是一個基於知識蒸餾的實驗，用於將大模型（教師模型）的知識轉移到小模型（學生模型）。
  - [Practice MNIST-distillation/](Practices/MNIST-distillation/)

- [CIFAR-10]()：CIFAR-10 是一個包含 60,000 張 32x32 像素的彩色圖像的數據集，分為 10 個類別。
  - [Practice CIFAR-10/](Practices/CIFAR-10/)

- [CIFAR-10]()：CIFAR-10 是一個包含 60,000 張 32x32 像素的彩色圖像的數據集，分為 10 個類別。
  - [Practice CIFAR-10/](Practices/CIFAR-10/)

- [Transformer-Decoder-only](Practice/Transformer-decoder-only/)：是一個基於 Transformer 的解碼器模型，用於生成文本（字）序列。

- [RAG]()：RAG 是一種基於檢索式生成模型的技術，可以透過檢索知識庫中的文本來生成回應。
  - [Practice RAG](Practices/RAG-demo/README.md)：RAG 模型 demo (中文，基於 BGE 模型，醫療領域)。

- [ ] [Multi-Modal from scratch]()
  - [ ] [SigLP]()

- [ ] [Large Language Model from scratch]()
  - [ ] [SFT]()
  - [ ] [DPO]()

- [ ] [Mixture of Experts (MoE) from scratch]()

- [ ] [LLM Knowledge Distillation]()

- [ ] [DeepSeek-R1 from scratch]()

## Application

- [dify-web-summarizer](Application/dify-web-summarizer/README.md)：一個簡易的Web 摘要瀏覽器插件應用實現，使用 Dify 來輔助建立的 AI 小工具。

- [browser-use-application](Application/browser-use-application/README.md)：基於 [browser-use](https://github.com/browser-use) 的測試應用。

- [MCP server applications]()
  - [simple-calculator-mcp](Application/simple-calculator-mcp/README.md)：一個簡單的計算機 MCP 應用。
  - control browser?
  - voice assistant?

## More News

- [OpenAI News](https://openai.com/news/)
- [Anthropic's research](https://www.anthropic.com/research)
- [Meta AI Blog](https://ai.meta.com/blog/)
- [Google AI Latest News](https://ai.google/latest-news/)
- [Hugging Face Blog](https://huggingface.co/blog)
- [The Rundown AI](https://www.therundown.ai/)
- [Sebastian Raschka's AI Magazine](https://magazine.sebastianraschka.com/?utm_source=homepage_recommendations&utm_campaign=1741130)
- [Maarten Grootendorst's Newsletter](https://newsletter.maartengrootendorst.com/)
- [Language Models Newsletter](https://newsletter.languagemodels.co/?utm_source=homepage_recommendations&utm_campaign=1741130)

