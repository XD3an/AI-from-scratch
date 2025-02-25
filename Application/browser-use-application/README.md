# browser-use

基於 [browser-use](https://github.com/browser-use) 的測試應用。

## Installation

1. create venv (uv) and install dependencies

    ```
    uv venv browser-venv --python 3.12 --seed    
    ```

2. activate venv

    ```
    browser-venv\Scripts\activate
    ```

## Usage

### [web-ui](https://github.com/browser-use/web-ui)

1. clone the repository

    ```
    git clone https://github.com/browser-use/web-ui
    ```

2. install dependencies

    ```
    cd web-ui
    pip install -r requirements.txt
    ```

3. run the app

    ```
    python webui.py --ip 127.0.0.1 --port 4444
    ```

### [browser-use](https://github.com/browser-use/browser-use)

- Based on Google Gen AI

#### browser-use command line (browser-use-cmd)

1. install dependencies

    ```
    pip install browser-use
    playwright install
    pip install -r requirements.txt
    ```

2. add api-key in your config file

3. run the app

    ```
    usage: browser-use-cmd.py [-h] --task TASK [--llm {google,openai,anthropic}] [--model MODEL] [--override-api-key API_KEY]
    # e.g. .\browser-use-cmd.py  --task "Search News by google and stop" --llm "google" --model "gemini-2.0-flash-001" 
    ```