import argparse
import logging
import sys

from dotenv import load_dotenv

# from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from tools import list_files, read_file

# ANSI escape codes for colored output
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "CYAN": "\033[96m",
    "ENDC": "\033[0m",
}


def color_print(color_key: str, *args, **kwargs):
    """Prints text in a specified color."""
    print(COLOR[color_key], end="")
    print(*args, **kwargs)
    print(COLOR["ENDC"], end="")


# def get_chat_model() -> BaseChatModel:
#     """
#     Initializes and returns a LangChain chat model.
#     This function localizes the provider-specific import and instantiation.
#     """
#     # The provider-specific import is contained within this function.
#     # from langchain_anthropic import ChatAnthropic

#     try:
#         # This is where you could add logic to switch between providers
#         # (e.g., based on an environment variable).
#         model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
#         return model
#     except Exception as e:
#         logging.error(f"Failed to initialize the ChatAnthropic model: {e}")
#         return None


def main():
    """Main function to set up and run the agent."""
    load_dotenv()  # Load environment variables from a .env file if it exists

    parser = argparse.ArgumentParser(
        description="Chat with a large language model using local tools."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # model = get_chat_model()
    model = init_chat_model(
        "claude-sonnet-4-5-20250929", temperature=0.5, timeout=10, max_tokens=1000
    )
    if not model:
        color_print("RED", "Error: Could not initialize the chat model.")
        print(
            "Please ensure the required model provider library (e.g., "
            "langchain-anthropic) is installed"
        )
        print(
            "and the necessary API key (e.g., ANTHROPIC_API_KEY) is set in your "
            "environment or a .env file."
        )
        sys.exit(1)

    # tools = [read_file, list_files, bash, edit_file]

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant."),
    #     ("placeholder", "{chat_history}"),
    #     ("human", "{input}"),
    #     ("placeholder", "{agent_scratchpad}"),
    # ])
    SYSTEM_PROMPT = """You are a helpful coding assistant.

    You have access to two tools:

    - read_file: use this to get the contents of a file
    - lkst_files: use this to list all files in a specific directory

    If a user asks you for file content, make sure you use the tools."""

    # Complete implementation of coding agent using latest Langchain 1.x API
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[read_file, list_files],
        # context_schema=Context,
        # response_format=str,
        checkpointer=checkpointer,
    )

    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What's in pyproject.toml?"}]},
        config=config,
        # context=Context(user_id="1")
    )

    print(response)

    first_ai_content = next(
        msg.content for msg in response["messages"] if isinstance(msg, ToolMessage)
    )

    print(first_ai_content)


if __name__ == "__main__":
    main()
