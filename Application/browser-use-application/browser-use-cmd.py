import os
import argparse
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import asyncio

# DEFAULT config
DEFAULT_API_KEY = "" 
DEFAULT_MODEL = ""

async def run_agent(task: str, model: str = DEFAULT_MODEL, api_key: str = None) -> str:
    """
    Run the agent with the specified task and model.

    Args:
        task (str): Task description for the agent.
        model (str): OpenAI model to use (default: DEFAULT_MODEL).
        api_key (str): OpenAI API key (optional, will use default if not provided).

    Returns:
        str: Result of the agent.
    """
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
    else:
        os.environ['GOOGLE_API_KEY'] = DEFAULT_API_KEY

    # Initialize the agent
    agent = Agent(
        task=task,
        llm=ChatGoogleGenerativeAI(model=model),  # Use Google Gemini
        browser=Browser(
            config=BrowserConfig(
                headless=False,
                disable_security=True
            )
        )
    )

    # Run the agent and return result
    return await agent.run()

def parse_arguments():
    """"Parse command line arguments for the agent."""
    parser = argparse.ArgumentParser(description='Browser Agent CLI')

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='Task description for the agent'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (optional, will use default if not provided)'
    )

    return parser.parse_args()

async def main():
    """main function to run the agent."""
    args = parse_arguments()

    try:
        # Run the agent
        result = await run_agent(
            task=args.task,
            model=args.model,
            api_key=args.api_key
        )
        print(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())