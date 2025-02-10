# MNIST

是著名的手寫數字圖像資料集，廣泛用於機器學習和深度學習模型的訓練和測試。

## Usage

### Environment

1. build the virtual environment

  ```
  python -m venv .venv
  
  # activate
  .venv/Scripts/activate     # Windows
  source .venv/bin/activate  # macOS and Linux
  
  # install required packages

  pip install -r requirements.txt

  ```

2. Run the practice
  1. 訓練教師模型

    ```
    python mnist-teacher-pt.py 
    ```
  2. 蒸餾技術至學生模型 (含蒸餾與無蒸餾比較)
    ```
    python mnist-student-pt.py
    ```

3. Stop the virtual environment

  ```
  decativate
  ```


