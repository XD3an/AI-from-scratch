import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import sys
import warnings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser, BrowserConfig
import asyncio

# DEFAULT config
DEFAULT_LLM_PROVIDER = "google"
DEFAULT_MODEL = "gemini-flask-2.0"
CONFIG_FILE = "config.json" 

def load_config() -> Dict[str, str]:
    """Load API keys from config file."""
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        # Create default config if it doesn't exist using environment variable names
        default_config = {
            "GOOGLE_API_KEY": "",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": ""
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    
    with open(config_path, 'r') as f:
        return json.load(f)

async def run_agent(task: str, llm_provider: str = DEFAULT_LLM_PROVIDER, 
                   model: str = DEFAULT_MODEL, api_key: str = None) -> str:
    """
    Run the agent with the specified task, LLM provider and model.

    Args:
        task (str): Task description for the agent.
        llm_provider (str): LLM provider to use (google, openai, anthropic).
        model (str): Model name to use.
        api_key (str): API key (optional, will use config if not provided).

    Returns:
        str: Result of the agent.
    """
    # Mapping of provider names to environment variable names
    env_vars = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    # Load API keys from config
    config = load_config()
    
    # Use provided API key or get from config using the environment variable name
    if api_key is None:
        env_var_name = env_vars[llm_provider]
        api_key = config.get(env_var_name, "")
    
    # Set environment variable for the chosen provider
    os.environ[env_vars[llm_provider]] = api_key
    
    # Initialize LLM based on provider
    if llm_provider == "google":
        llm = ChatGoogleGenerativeAI(model=model)
    elif llm_provider == "openai":
        llm = ChatOpenAI(model=model)
    elif llm_provider == "anthropic":
        llm = ChatAnthropic(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Initialize browser separately for proper cleanup
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True
        )
    )
    
    # Initialize the agent
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser
    )

    try:
        # Run the agent and return result
        return await agent.run()
    finally:
        # Ensure browser resources are properly cleaned up
        try:
            await browser.close()
        except Exception:
            pass

def parse_arguments():
    """Parse command line arguments for the agent."""
    parser = argparse.ArgumentParser(
        description='Browser Agent CLI - Uses config.json for API keys by default'
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task description for the agent'
    )

    parser.add_argument(
        '--llm',
        type=str,
        default=DEFAULT_LLM_PROVIDER,
        choices=['google', 'openai', 'anthropic'],
        help=f'LLM provider to use (default: {DEFAULT_LLM_PROVIDER})'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        '--override-api-key',
        type=str,
        default=None,
        dest='api_key',
        help='Override the API key from config.json (optional)'
    )

    return parser.parse_args()

async def main():
    """main function to run the agent."""
    args = parse_arguments()

    try:
        # Run the agent
        result = await run_agent(
            task=args.task,
            llm_provider=args.llm,
            model=args.model,
            api_key=args.api_key  # Will be None by default, triggering config.json usage
        )
        print(f"Agent result: {result}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())