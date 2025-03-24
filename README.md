# AI from scratch

## AI Fundamental

### Tranditional Machine Learning Fundamental

<details>
<summary>Regression methods</summary>

- **Linear Regression（線性迴歸）**：預測連續值的最基本方法，尋找數據之間的線性關係。
- **Polynomial Regression（多項式迴歸）**：當數據關係非線性時使用，透過添加次方項來擬合曲線。
- **Ridge/Lasso Regression（嶺迴歸/套索迴歸）**：處理多重共線性問題的正則化線性迴歸方法，透過懲罰項來減少過擬合。
- **Support Vector Regression (SVR)（支持向量迴歸）**：利用支持向量找到最大容忍誤差的函數，可處理非線性關係。
- **Decision Tree Regression（決策樹迴歸）**：基於樹狀結構進行預測，將數據分割成不同區域。
- **Random Forest Regression（隨機森林迴歸）**：結合多個決策樹的集成學習方法，提高穩定性和準確性。

</details>

<details>
<summary>Classifications & Clustering methods</summary>

- **Logistic Regression（邏輯迴歸）**：用於二分類問題的統計方法，預測事件發生的概率。
- **Support Vector Machines (SVM)（支持向量機）**：在特徵空間中找尋最佳分隔超平面來分類數據。
- **Decision Trees（決策樹）**：樹狀結構模型，根據特徵值進行分支決策。
- **Random Forests（隨機森林）**：多個決策樹的集成，減少過擬合風險。
- **Naive Bayes（樸素貝葉斯）**：基於貝葉斯定理的分類器，假設特徵之間相互獨立。
- **K-Nearest Neighbors (KNN)（K近鄰）**：基於相似性的分類方法，根據最近的K個鄰居進行分類。
- **K-Means Clustering（K均值聚類）**：將數據分成K個不同的群組，最小化群內點到中心的距離和。
- **Hierarchical Clustering（層次聚類）**：創建數據點的嵌套聚類，可以是自下而上或自上而下的方法。
- **DBSCAN（基於密度的聚類方法）**：基於密度的聚類演算法，能發現任意形狀的聚類，對噪聲數據處理良好。
- **Gradient Boosting Machines (GBM)（梯度提升機）**：一種集成方法，通過構建弱學習器序列來提高性能。
- **XGBoost, LightGBM, CatBoost**：梯度提升的高效實現，針對不同場景優化的算法。

</details>

<details>
<summary>Dimensionality Reduction methods</summary>

- **Principal Component Analysis (PCA)（主成分分析）**：通過找出最大方差方向將高維數據投影到低維空間。
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)（t分布隨機鄰域嵌入）**：非線性降維技術，特別適合可視化高維數據。
- **Linear Discriminant Analysis (LDA)（線性判別分析）**：監督式降維技術，尋找能最佳區分不同類別的投影方向。
- **Singular Value Decomposition (SVD)（奇異值分解）**：矩陣分解技術，用於壓縮和降噪。
- **Non-negative Matrix Factorization (NMF)（非負矩陣分解）**：將矩陣分解為非負因子，適用於特徵提取。
- **Uniform Manifold Approximation and Projection (UMAP)（均勻流形近似與投影）**：現代非線性降維技術，保留全局和局部結構。
- **Autoencoders（自編碼器）**：神經網路架構，通過學習數據的壓縮表示實現降維。

</details>

### Deep Learning Fundamental

<details>
<summary>Multilayer Perceptron (MLP)（多層感知器）</summary>
最基本的前饋神經網路，由輸入層、一個或多個隱藏層和輸出層組成。每個神經元與下一層的所有神經元相連，用於分類和回歸任務。
</details>

<details>
<summary>Convolutional Neural Network (CNN)（卷積神經網路）</summary>
專為處理網格狀數據（如圖像）設計的神經網路。使用卷積操作自動提取特徵，通過卷積層、池化層和全連接層構成，在計算機視覺領域表現優異。
</details>

<details>
<summary>Recurrent Neural Network (RNN)（循環神經網路）</summary>
設計用於處理序列數據的神經網路。網絡中的神經元可以記住之前的訊息，適合處理時間序列、文本等序列數據。
</details>

<details>
<summary>Long Short-Term Memory (LSTM)（長短期記憶網路）</summary>
RNN的一種變體，解決了普通RNN的長期依賴問題。通過引入門控機制（輸入門、遺忘門、輸出門）來控制訊息流，有效處理長序列數據。
</details>

<details>
<summary>Variational Autoencoder (VAE)（變分自編碼器）</summary>
生成模型的一種，結合了自編碼器和機率模型。透過學習數據的隱變量表示，可以生成新的數據樣本，廣泛應用於圖像生成和特徵學習。
</details>

<details>
<summary>Generative Adversarial Networks (GAN)（生成對抗網路）</summary>
由生成器和判別器組成的對抗性架構。生成器嘗試創建逼真的數據，判別器嘗試區分真實和生成的數據，二者相互競爭提升。應用於圖像生成、風格轉換等領域。
</details>

<details>
<summary>Transformer</summary>
一種基於自注意力機制的神經網路架構，最初為自然語言處理設計。不依賴循環結構，通過注意力機制直接建模長距離依賴關係，展現出卓越的並行處理能力。是現代大語言模型（如GPT、BERT）的基礎架構。
  - [Transformer-Decoder-only](Practice/Transformer-decoder-only/README.md)
</details>

</details>

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

- [Anthropic's prompt library](https://docs.anthropic.com/en/prompt-library/library)

##### Workflow

- [n8n](https://github.com/n8n-io/n8n)：是一個開源的工作流自動化工具，可以用於構建和執行機器學習工作流。

- [dify](https://github.com/langgenius/dify)：是一個用於構建和部署大型語言模型（LLM）的工作流自動化工具，支持多種不同類型的模型和任務。

##### Agents

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)：MCP 是一個用於定義和交換模型上下文的協議，用於描述模型的環境、任務和目標，以及模型的訓練和推理過程。
  - [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
  - [MCP servers githuModel Context Protocol servers](https://github.com/modelcontextprotocol/servers)
  - [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)：是官方及社區建立的MCP servers 集合。
  - [PulseMCP](https://www.pulsemcp.com/)
  - [MCP.so](https://mcp.so/)

  - More: [XD3an/model-context-protocol-templates](https://github.com/XD3an/model-context-protocol-templates)

- [Github Copilot](https://github.com/features/copilot)：微軟和OpenAI合作開發的AI編程助手，可在多種集成開發環境中提供代碼建議和自動完成。支持各種編程語言，能夠理解上下文並提供相關代碼片段。

- [Cline](https://www.cline.bot/)：Cline 提供了一個靈活且可擴展的框架，支持多種 AI 模型和 MCP 功能，幫助開發者快速構建和部署智慧代理系統。
  - [Roo-Code](https://github.com/RooVetGit/Roo-Code)

- [Continue](https://www.continue.dev/)：專為開發者設計的AI助手框架，提供自定義AI代碼幫助，通過IDE擴展和模型、規則、提示等構建塊整合。

- [Deep Research]()：是一種利用 AI agent 來自動執行多步驟網路研究的功能。讓 AI 從網路上自動抓取並分析大量資料（例如文本、圖像、PDF等），然後進行邏輯推理與綜合，最終生成一份類似於專業研究分析師所撰寫的詳盡報告。
  - [node-DeepResearch](https://github.com/jina-ai/node-DeepResearch)

- [browser-use](https://github.com/browser-use/browser-use)：一個用於使 AI 代理可以訪問網頁的工具。

- [crawl4ai](https://github.com/unclecode/crawl4ai)：用於幫助 AI 爬取相關網站的工具。

- [huggingface microsoft/OmniParser-v2.0 · Hugging Face](https://huggingface.co/microsoft/OmniParser-v2.0?fbclid=IwY2xjawIe5rVleHRuA2FlbQIxMQABHVOSUSGV78Bt5i6EoNQ-WEDWtFBpQYlz7nDE4BQlWJHYCpKFt-Fl9KWZVQ_aem_Um2hOeKCFYrCZZr35fedxw
)：是一個用於解析和理解文本的工具，可以幫助 AI 代理更好地理解和處理文本資料。

- [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)：是一個首創的可擴展真實電腦環境，專為評估多模態AI代理在開放式任務中的能力而設計。這個環境可以讓AI代理與真實的作業系統（如Ubuntu、Windows和macOS）進行交互，執行各種複雜的電腦任務。

- [geekan/MetaGPT](https://github.com/geekan/MetaGPT)：多代理框架，模擬軟體公司的運作方式，能夠從一行需求輸入生成用戶故事、競爭分析、需求、結構、API等完整軟體開發流程。

- [OpenManus](https://github.com/mannaandpoem/OpenManus)：是一個開源框架，用於構建通用 AI 代理。該項目旨在提供一個無需邀請碼的 Manus 替代方案，讓開發者能夠自由實現各種 AI 代理相關的想法。由 MetaGPT 團隊成員（@Xinbin Liang、@Jinyu Xiang 等）在短短 3 小時內推出原型，並持續開發中。

- [AutoGen](https://github.com/microsoft/autogen)：微軟開發的框架，用於創建能夠自主行動或與人類協作的多代理AI應用。

- [LangGraph](https://github.com/langchain-ai/langgraph)：LangGraph 是一個用於構建和部署多代理 AI 系統的框架，支持多種不同類型的代理，包括語言模型、知識庫、對話系統等。

- [openai-agents-python](https://github.com/openai/openai-agents-python)：是由 OpenAI 開發的輕量級但功能強大的框架，專為構建多代理工作流程而設計。這個開源框架提供了一套簡潔的工具，使開發者能夠輕鬆創建、部署和管理 AI 代理，特別適合需要多個代理協同工作的複雜應用場景。

- [octotools](https://github.com/octotools/octotools)：是一個無需訓練、友好且易於擴展的開源代理框架，專為解決跨多個領域的複雜推理問題而設計。該框架引入了標準化的工具卡片來封裝工具功能，一個處理高低層次規劃的規劃器，以及用於執行工具使用的執行器。

- [AutoAgent](https://github.com/HKUDS/AutoAgent)：AutoAgent 是一個全自動且高度自我發展的框架，使用戶能夠僅通過自然語言創建和部署 LLM 代理。該項目提供了一種革命性的方法，讓使用者不需要編寫一行代碼就能構建複雜的 AI 代理應用。

- [PocketFlow](https://github.com/The-Pocket/PocketFlow)：PocketFlow 是一個僅有 100 行代碼的極簡主義 LLM 框架，以「讓代理構建代理」為核心理念。它旨在提供一個無膨脹、無依賴、無供應商鎖定的輕量級解決方案，同時保持足夠的表達能力來實現複雜的 AI 應用功能。

- [bytedance/UI-TARS-desktop](https://github.com/bytedance/UI-TARS-desktop)

### Tools and Repositories

#### AI Tools

- 基礎大語言模型
  
  - [ChatGPT](https://chat.openai.com/)：ChatGPT 是由 OpenAI 開發的對話式大型語言模型，自2022年11月推出以來，迅速成為全球最廣泛使用的 AI 對話產品之一。
  
  - [Claude](https://claude.ai/)：Claude 是由 Anthropic 開發的對話式 AI 助手，以安全性和有幫助性為設計理念。

  - [Gemini](https://gemini.google.com/app)：Gemini 是 Google 開發的多模態大語言模型系列。

  - [xAI](https://x.ai/)：Grok 是由 xAI 公司開發的對話式 AI 模型，旨在提供更直接、有時甚至帶有幽默感的回應。

  - [Meta Llama Llama](https://www.llama.com/)： 是由 Meta（前 Facebook）開發的開源大型語言模型系列，特點是提供開源代碼供研究和商業應用。

  - [DeepSeek](https://www.deepseek.com/)：是由中國團隊開發的大語言模型系列，致力於提供多語言能力和先進的技術能力。

  - ...

- AI 搜尋

  - [Perplexity AI](https://www.perplexity.ai/)：一個 AI 驅動的搜尋引擎，能夠為複雜問題提供詳盡且附帶引用來源的答案。它結合了網路搜尋功能和大型語言模型，生成全面的回答並提供引用。特點包括實時信息訪問、多步驟推理和處理複雜查詢的能力。

- AI 畫圖

  - [midjourney](https://www.midjourney.com/)：是一款流行的 AI 圖像生成器，能從文字提示創建高質量、藝術風格的圖像。它以藝術風格和美學質量著稱，在寫實和風格化插圖方面表現尤為出色。用戶主要通過 Discord 與其互動，使用文字命令生成圖像。它提供各種設置來控制圖像風格、質量和縱橫比。

  - [Stability AI](https://stability.ai/)：是 Stable Diffusion（一款領先的開源 AI 圖像生成模型）背後的公司。他們提供各種圖像生成功能，包括用於生成高品質圖像的 Stable Diffusion XL。產品範圍涵蓋圖像、視頻、音頻和 3D 內容生成工具。公司為開發者和企業提供商業授權選項和自託管部署方案。

  - [Flux AI](https://flux1.ai/)：是一款以高質量輸出和精準遵循提示詞著稱的 AI 圖像生成器。由 BlackForestLabs 開發，提供多個版本包括 Flux.1 Pro、Dev 和 Schnell。該平台支持生成最高 2.0 百萬像素的高分辨率圖像，並提供各種縱橫比選項。它使用擴展到 12B 參數的先進基於 transformer 的流量模型，以獲得卓越的圖像質量。

  - [whisk](http://labs.google/whisk)：是 Google Labs 開發的文字轉圖像生成器，專為創建具有一致風格的藝術圖像而設計。它以高質量圖像生成聞名，注重匹配輸入提示的風格和內容。與一些競爭對手不同，它旨在產生更連貫和美學上令人愉悅的結果。它由 Google 的研究和實驗室團隊開發，利用他們在 AI 和圖像處理方面的專業知識。

  - [Recraft](https://www.recraft.ai/)：是一個面向專業設計師的設計導向 AI 圖像和向量生成平台。它提供工具來生成和編輯 AI 圖像，對風格和構圖進行精確控制。功能包括 AI 圖像生成、矢量化、模型生成和無縫圖案創建。設計師可以通過風格存儲和共享功能維持品牌一致性。

- AI 影片

  - [Sora](https://openai.com/sora/)： 是 OpenAI 的文字到影片 AI 模型，能夠根據文字指令生成逼真和富有想像力的場景。它可以創建長達 60 秒的影片，包含複雜場景、多個角色、特定類型的動作和準確的細節。該模型不僅理解場景中有什麼，還理解物體在物理世界中如何移動和互動。目前處於研究預覽階段，存取有限。

  - [Runway](https://runwayml.com/research/gen-2)：Runway 的 Gen-2 是一個多模態 AI 系統，可以從文字、圖像或視頻剪輯生成影片。它支持多種生成模式：文字到視頻、圖像到影片、影片風格化等。該平台專為電影製作人和創作者設計，無需傳統拍攝即可生成逼真的影片內容。用戶可以將圖像的風格應用到影片中，並為特定視覺美學創建自定義模型。

  - [Pika](https://pika.art/)：是一個 AI 影片生成平台，可將文字、圖像和視頻剪輯轉換為新影片。其最新版本 Pika 2.2 包括 Pikaframes 等功能，用於創建照片之間的過渡效果。該平台允許用戶透過簡單的文字提示生成、編輯和增強影片。它被定位為一個無需傳統影片編輯技能即可進行創意影片製作的易用工具。

- AI 語音

  - [ElevenLabs](https://elevenlabs.io/)：ElevenLabs 是一家領先的 AI 語音生成平台，以創建高度逼真和自然的聲音而聞名。它提供語音克隆技術、多語言文字轉語音和可自定義的語音風格和情感。該平台用於有聲書、配音、內容創作和對話式 AI 應用。它支持超過 30 種語言，並為開發者提供網頁界面和 API 存取。

  - [ChatTTS](https://chattts.com/)：是一個專為對話場景優化的文字轉語音模型。它專為語言模型助手的對話任務和會話應用而設計。該系統支持包括英語和中文在內的多種語言，並具有自然的語調。它經過大約 100,000 小時的語音數據訓練，以實現高質量和自然度。

- AI 音樂

  - [Suno](https://suno.com/)：是一個 AI 音樂生成平台，可以從文字提示創建完整的歌曲。它能夠生成帶有人聲、樂器和製作質量的原創音樂，這些音樂聽起來驚人地專業。用戶可以在提示中指定流派、情緒、風格和歌詞內容。該平台發展迅速，其最新模型能夠製作越來越復雜的音樂作品。

- AI PPT

  - [Gamma](https://gamma.app/)： 是一個 AI 驅動的演示和文檔創作平台。它允許用戶從簡單的提示或大綱創建專業的演示文稿、文檔和網頁。該平台結合了 AI 內容生成和設計功能，以產生視覺吸引力的演示文稿。功能包括自動格式化、設計建議以及從簡單項目符號擴展內容的能力。

- AI Coding

  - [Github Copilot](https://github.com/features/copilot)

  - [Cline](https://www.cline.bot/)

  - [Continue](https://www.continue.dev/)

  - [Cursor](https://www.cursor.com/)
    - 核心功能：
      - Rules
      - Composer
      - Chat
      - Edit: Inline & Terminal & Tabs
      - MCP
      - Agents
      - ...
    - 參考資料：
      - [cursor.directory](https://cursor.directory/)

  - [Windsurf](https://codeium.com/windsurf)

  - [Claude Code](https://github.com/anthropics/claude-code)
    - [claude-code overview](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)

- AI Agent

  - [manus](https://manus.im/)：是由中國初創公司 Monica 開發的自主人工智慧代理（AI Agent），於 2025 年 3 月 6 日正式推出。 

- 其他實用工具

  - [Monica](https://monica.im/)： 是一個多功能AI助理，整合了多個頂尖AI模型，包括OpenAI的o3-mini、DeepSeek R1、GPT-4o、Claude 3.7和Gemini 2.0。它提供全方位的AI工具套件，包括聊天、摘要、寫作、搜索、翻譯、圖像生成等多種功能。

  - [NotebookLm](https://notebooklm.google.com/)：是Google開發的AI驅動筆記和研究工具。它允許用戶上傳檔案並創建AI輔助的"筆記本"，能夠分析、總結並回答關於內容的問題。

  - [沉浸式翻譯](https://immersivetranslate.com/)：是一款高評價的雙語翻譯瀏覽器擴展，提供網頁、PDF、EPUB和視頻字幕翻譯功能。它支持超過10種翻譯服務，包括OpenAI (ChatGPT)、DeepL和Gemini。

  - [HeyGen](https://www.heygen.com/)：是一個AI影片生成平台，專門創建專業級AI頭像和影片。它允許用戶生成逼真的影片演示，配有可自定義的AI頭像，可以用多種語言和聲音說話。

#### Collections

- [awesome-ai](https://github.com/openbestof/awesome-ai)：收集各種 AI 相關的資源和工具。

- [ai-collection](https://github.com/ai-collection/ai-collection)：一個 AI 工具和專案的集合。

- [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)：收集各種大型語言模型（LLM）的推理工具和框架。

- [Awesome-LLM-resources](https://github.com/WangRongsheng/awesome-LLM-resourses)：收集各種大型語言模型（LLM）的相關資源。

- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)：Model Context Protocol (MCP) servers awesome list.

- [Awesome-AI-Agents](https://github.com/Jenqyang/Awesome-AI-Agents)：收集各種 AI-Agents 相關資訊。

- [awesome-computer-use](https://github.com/ranpox/awesome-computer-use)：收集了各種用於幫助 AI 代理訪問網頁、爬取數據、解析文本等的工具。

- [awesome-assistants](https://github.com/awesome-assistants/awesome-assistants)

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

