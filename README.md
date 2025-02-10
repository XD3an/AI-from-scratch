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

- [Transformer-Decoder-only](Transformer-decoder-only/README.md)

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

- [HuggingFace-Transformers](https://huggingface.co/docs/transformers/en/index): HuggingFace-transformers 是一個用於自然語言處理的庫，提供了許多預訓練的語言模型，包括 BERT、GPT-2、RoBERTa 等。HuggingFace-transformers 提供了一個簡單而強大的 API，可以用於構建和訓練自然語言處理模型，並且支持多種不同的任務，包括文本分類、命名實體識別、問答等。
  - [gpt2-inference/](HuggingFace-Transformers/)

- [vLLM](https://docs.vllm.ai/en/latest/index.html): vLLM 是一個開源的高效能推理框架，專門針對大規模語言模型（LLMs）在 GPU 上的推理進行優化。其目的是提高大模型推理的速度和效率，特別是在處理大規模神經網路模型時。

- [FastAPI + Transformers]()
  - [FastAPI](https://fastapi.tiangolo.com/tutorial/): FastAPI 是一個現代的 Web 框架，用於構建高性能的 API。FastAPI 提供了一個簡單而強大的 API。

## Practices

- [MNIST](): MNIST 是一個手寫數字圖像數據集，包含 0 到 9 的 70,000 張 28x28 像素的灰度圖像。
  - [Practice MNIST/](Practices/MNIST/)

- [MNIST-distillation](): MNIST-distillation 是一個基於知識蒸餾的實驗，用於將大模型（教師模型）的知識轉移到小模型（學生模型）。
  - [Practice MNIST-distillation/](Practices/MNIST-distillation/)

- [CIFAR-10](): CIFAR-10 是一個包含 60,000 張 32x32 像素的彩色圖像的數據集，分為 10 個類別。
  - [Practice CIFAR-10](Practices/CIFAR-10)

- ...

## More News

- [OpenAI News](https://openai.com/news/)
- [Meta AI Blog](https://ai.meta.com/blog/)
- [Google AI Latest News](https://ai.google/latest-news/)
- [Hugging Face Blog](https://huggingface.co/blog)
- [Sebastian Raschka's AI Magazine](https://magazine.sebastianraschka.com/?utm_source=homepage_recommendations&utm_campaign=1741130)
- [Maarten Grootendorst's Newsletter](https://newsletter.maartengrootendorst.com/)
- [Language Models Newsletter](https://newsletter.languagemodels.co/?utm_source=homepage_recommendations&utm_campaign=1741130)

