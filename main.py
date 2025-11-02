import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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


# --- Tool Definitions using Pydantic and @tool decorator ---

class ReadFileInput(BaseModel):
    path: str = Field(description="The relative path of a file in the working directory.")

@tool(args_schema=ReadFileInput)
def read_file(path: str) -> str:
    """Read the contents of a given relative file path."""
    logging.info(f"Reading file: {path}")
    try:
        content = Path(path).read_text()
        logging.info(f"Successfully read file {path} ({len(content)} bytes)")
        return content
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return f"Error: File not found at '{path}'"
    except Exception as e:
        logging.error(f"Failed to read file {path}: {e}")
        return f"Error: Failed to read file '{path}': {e}"

class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="Optional relative path to list files from. Defaults to current directory.")

@tool(args_schema=ListFilesInput)
def list_files(path: str = ".") -> str:
    """List files and directories at a given path."""
    logging.info(f"Listing files in directory: {path}")
    try:
        base_path = Path(path)
        files = []
        for p in sorted(base_path.rglob("*")):
            # Skip hidden directories like .devenv or .git
            if any(part.startswith('.') for part in p.parts):
                continue
            rel_path = p.relative_to(base_path)
            files.append(str(rel_path) + ("/" if p.is_dir() else ""))
        logging.info(f"Successfully listed {len(files)} files in {path}")
        return json.dumps(files)
    except Exception as e:
        logging.error(f"Failed to list files in {path}: {e}")
        return f"Error: Failed to list files in '{path}': {e}"

class BashInput(BaseModel):
    command: str = Field(description="The bash command to execute.")

@tool(args_schema=BashInput)
def bash(command: str) -> str:
    """Execute a bash command and return its output."""
    logging.info(f"Executing bash command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )
        output = result.stdout.strip()
        logging.info(
            f"Bash command executed successfully, output length: {len(output)} chars"
        )
        return output
    except subprocess.CalledProcessError as e:
        logging.error(f"Bash command failed: {e}")
        return f"Command failed with exit code {e.returncode}:\n{e.stderr}"

class EditFileInput(BaseModel):
    path: str = Field(description="The path to the file.")
    old_str: str = Field(description="The exact text to search for and replace.")
    new_str: str = Field(description="The text to replace old_str with.")

@tool(args_schema=EditFileInput)
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """
    Make edits to a text file by replacing content.

    This tool replaces the first occurrence of 'old_str' with 'new_str' in the
    specified file. If 'old_str' is an empty string, 'new_str' is appended to the file.
    If the file doesn't exist and 'old_str' is empty, the file will be created.
    """
    if not path or old_str == new_str:
        logging.error("EditFile failed: invalid input parameters")
        return "Error: Invalid input parameters. Path cannot be empty and old_str must differ from new_str."

    file_path = Path(path)
    logging.info(f"Editing file: {path} (replacing {len(old_str)} chars with {len(new_str)} chars)")

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists() and old_str == "":
            logging.info(f"File does not exist, creating new file: {path}")
            file_path.write_text(new_str)
            return f"Successfully created and wrote to new file {path}"

        old_content = file_path.read_text()

        if old_str == "":
            new_content = old_content + new_str
        else:
            count = old_content.count(old_str)
            if count == 0:
                return "Error: old_str not found in file."
            if count > 1:
                return f"Error: old_str found {count} times, must be unique for safety."
            new_content = old_content.replace(old_str, new_str, 1)

        file_path.write_text(new_content)
        logging.info(f"Successfully edited file {path}")
        return "OK"
    except Exception as e:
        logging.error(f"Failed to edit file {path}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def get_chat_model() -> BaseChatModel:
    """
    Initializes and returns a LangChain chat model.
    This function localizes the provider-specific import and instantiation.
    """
    # The provider-specific import is contained within this function.
    from langchain_anthropic import ChatAnthropic
    try:
        # This is where you could add logic to switch between providers
        # (e.g., based on an environment variable).
        model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        return model
    except Exception as e:
        logging.error(f"Failed to initialize the ChatAnthropic model: {e}")
        return None


def main():
    """Main function to set up and run the agent."""
    load_dotenv()  # Load environment variables from a .env file if it exists

    parser = argparse.ArgumentParser(description="Chat with a large language model using local tools.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    model = get_chat_model()
    if not model:
        color_print("RED", "Error: Could not initialize the chat model.")
        print("Please ensure the required model provider library (e.g., langchain-anthropic) is installed")
        print("and the necessary API key (e.g., ANTHROPIC_API_KEY) is set in your environment or a .env file.")
        sys.exit(1)

    tools = [read_file, list_files, bash, edit_file]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Complete implementation of coding agent using latest Langchain 1.x API


if __name__ == "__main__":
    main()


