# AI from scratch

## Machine Learning Fundamental

- **Regression methods**

- **Classifications & Clustering methods**

- **Dimensionality Reduction methods**

## Deep Learning Fundamental

- [ ] Multilayer perceptron (MLP)

- [ ] Convolution Neural Network (CNN)

- [ ] Recurrent Neural Network (RNN)

- [ ] Long short-term memory (LSTM)

- [ ] Variational autoencoder (VAE)

- [ ] Generative adversarial networks (GAN)

- Transformer

  - [Transformer-Decoder-only](Practice/Transformer-decoder-only/README.md)

## AI Framework and Ecosystem

### Deep Learning Frameworks

- [TensorFlow](https://www.tensorflow.org/guide): TensorFlow 是一個開源的機器學習框架，由 Google Brain 團隊開發。提供了一個用於構建和訓練深度神經網路的 API，並且支持多種硬體加速器，包括 GPU 和 TPU。TensorFlow 提供了一個靈活的計算圖模型，可以用於構建各種不同類型的機器學習模型，包括卷積神經網路、循環神經網路、自編碼器、生成對抗網路等。

- [PyTorch](https://pytorch.org/get-started/locally/): PyTorch 是一個開源的深度學習框架，由 Facebook 的 AI 研究團隊開發。提供了一個用於構建和訓練深度神經網路的 API，並且支持 GPU 和 TPU 加速。

- [Keras](https://keras.io/): Keras 是一個深度學習框架，可以運行在 TensorFlow、Theano 和 CNTK 等後端。Keras 提供了一個簡單而直觀的 API，可以用於構建和訓練深度神經網路。Keras 的設計目標是讓用戶能夠快速地構建和訓練深度學習模型，而不需要深入了解底層的實現細節。

### Fine-tuning

- [unsloth](https://github.com/unslothai/unsloth): 一個用於微調深度學習模型的工具，可以用於微調預訓練模型，並且支持多種不同類型的前沿模型。

### Model Compression

- [Quantization](): 量化是透過將模型權重和運算從高精度（例如 FP32）轉換為低精度（例如 INT8、BF16）來減少計算資源需求。

- [Pruning](): 剪枝技術移除影響較小的權重或神經元，以減少模型規模並加快運算速度。

- [Knowledge Distillation](): 知識蒸餾透過讓小模型（學生模型）學習大模型（教師模型）的輸出來獲得相似的性能。

- [Binarization](): 將模型權重和激活值約束為 0 或 1，以極大減少計算需求（例如 XNOR-Net）。

### Inference and Deployment

#### Inference

- [WebLLM](https://webllm.mlc.ai/): 是一個用於在瀏覽器中運行大型語言模型（LLMs）的推理引擎工具，可以用於生成文本、回答問題等。

- [LMStudio](https://lmstudio.ai/): 是一個功能齊全的本地部屬 LLM 運行環境框架，支援本地設備上離線執行大型語言模型。

- [Ollama](https://ollama.com/): 是一個開源的 LLM 服務框架，用於本地部署，且提供完整的模型管理與推理服務，適用於對資料安全性要求較高的應用環境。

- [vLLM](https://docs.vllm.ai/en/latest/index.html): vLLM 是一個開源的高效能推理框架，專門針對大規模語言模型（LLMs）在 GPU 上的推理進行優化。其目的是提高大模型推理的速度和效率，特別是在處理大規模神經網路模型時。

- [LightLLM](https://github.com/ModelTC/lightllm): 是一個基於 Python 的輕量級大型語言模型 (LLM) 推理和服務框架。它設計上註重高效能與低資源消耗，適用於在資源受限的環境中（如行動裝置或邊緣運算）快速部署 LLM。

- [OpenLLM](https://github.com/bentoml/OpenLLM): 允許開發人員使用單一命令運行任何開源 LLM（Llama 3.3、Qwen2.5、Phi3 等）或自訂模型作為與 OpenAI 相容的 API。

- [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/en/index): 是一個用於部署和服務大型語言模型（LLM）的工具包。
    - [HuggingFace-Transformers/gpt2-inference/](https://www.notion.so/AI%20Framework%20and%20Ecosystem/HuggingFace-Transformers/)

- [GPT4ALL](https://www.nomic.ai/gpt4all): 是一個旨在讓大型語言模型（LLM）在本地設備上運行的開源平台。

- [llama.cpp](https://github.com/ggml-org/llama.cpp): 是一個開源的 C/C++ 函式庫，旨在在各種硬體上有效率地進行大型語言模型（LLM）的推理。

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): 是 NVIDIA 開源的 TensorRT 工具箱，專門用於優化大型語言模型（LLM）的推理效能。它提供 Python API 和 C++ 元件，允許使用者有效地在 NVIDIA GPU 上建置和執行 TensorRT 推理引擎。

#### Interactions

##### Prompt

- [Prompting Guide](https://www.promptingguide.ai/)

- [Promptify](https://app.promptify.com/)

- [AIDungeon](https://aidungeon.com/)

- [Promptbase](https://promptbase.com/)

## Agents

- [MCP protocol]()

## Practices

- [MNIST](): MNIST 是一個手寫數字圖像數據集，包含 0 到 9 的 70,000 張 28x28 像素的灰度圖像。
  - [Practice MNIST/](Practices/MNIST/)

- [MNIST-distillation](): MNIST-distillation 是一個基於知識蒸餾的實驗，用於將大模型（教師模型）的知識轉移到小模型（學生模型）。
  - [Practice MNIST-distillation/](Practices/MNIST-distillation/)

- [CIFAR-10](): CIFAR-10 是一個包含 60,000 張 32x32 像素的彩色圖像的數據集，分為 10 個類別。
  - [Practice CIFAR-10/](Practices/CIFAR-10/)

- [CIFAR-10](): CIFAR-10 是一個包含 60,000 張 32x32 像素的彩色圖像的數據集，分為 10 個類別。
  - [Practice CIFAR-10/](Practices/CIFAR-10/)

- [Transformer-Decoder-only](Practice/Transformer-decoder-only/): 是一個基於 Transformer 的解碼器模型，用於生成文本（字）序列。

- [RAG]: RAG 是一種基於檢索式生成模型的技術，可以透過檢索知識庫中的文本來生成回應。
  - [Practice RAG](Practices/RAG-demo/README.md): RAG 模型 demo (中文，基於 BGE 模型，醫療領域)。

- [Multi-Modal from scratch]
  - [SigLP]

- [Large Language Model from scratch]
  - [SFT]
  - [DPO]

- [MoE]

- [LLM Knowledge Distillation]

- [DeepSeek-R1 from scratch]

## Application

- [dify-web-summarizer](Application/dify-web-summarizer): 一個簡易的Web 摘要瀏覽器插件應用實現，使用 Dify 來輔助建立的 AI 小工具。

- [mcp-servers]

- [Medical AI applications]

### Tools and Repositories

#### Collections

- [awesome-ai](https://github.com/openbestof/awesome-ai): 收集各種 AI 相關的資源和工具。

- [ai-collection](https://github.com/ai-collection/ai-collection): 一個 AI 工具和專案的集合。

- [dify](https://github.com/langgenius/dify): 一個支持 LLM 應用開發的開源框架。

- [crawl4ai](https://github.com/unclecode/crawl4ai): 用於爬取 AI 相關數據的工具。

- [browser-use](https://github.com/browser-use/browser-use): 一個用於使 AI 代理可以訪問網頁的工具。

- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers): Model Context Protocol (MCP) servers awesome list.

- [Awesome-AI-Agents](https://github.com/Jenqyang/Awesome-AI-Agents): 收集各種 AI-Agents 相關資訊。

- [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)

- [Awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resourses)

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

## More News

- [OpenAI News](https://openai.com/news/)
- [Meta AI Blog](https://ai.meta.com/blog/)
- [Google AI Latest News](https://ai.google/latest-news/)
- [Hugging Face Blog](https://huggingface.co/blog)
- [The Rundown AI](https://www.therundown.ai/)
- [Sebastian Raschka's AI Magazine](https://magazine.sebastianraschka.com/?utm_source=homepage_recommendations&utm_campaign=1741130)
- [Maarten Grootendorst's Newsletter](https://newsletter.maartengrootendorst.com/)
- [Language Models Newsletter](https://newsletter.languagemodels.co/?utm_source=homepage_recommendations&utm_campaign=1741130)

