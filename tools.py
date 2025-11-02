"""Tool definitions for the codegen demo agent."""

import json
import logging
import subprocess
from pathlib import Path

from langchain.tools import tool

from pydantic import BaseModel, Field


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
    path: str = Field(
        default=".",
        description="Optional relative path to list files from. Defaults to current directory.",
    )


@tool(args_schema=ListFilesInput)
def list_files(path: str = ".") -> str:
    """List files and directories at a given path."""
    logging.info(f"Listing files in directory: {path}")
    try:
        base_path = Path(path)
        files = []
        for p in sorted(base_path.rglob("*")):
            # Skip hidden directories like .devenv or .git
            if any(part.startswith(".") for part in p.parts):
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
    """Make edits to a text file by replacing content."""
    if not path or old_str == new_str:
        logging.error("EditFile failed: invalid input parameters")
        return (
            "Error: Invalid input parameters. Path cannot be empty and old_str must differ"
            " from new_str."
        )

    file_path = Path(path)
    logging.info(
        f"Editing file: {path} (replacing {len(old_str)} chars with {len(new_str)} chars)"
    )

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
                return (
                    f"Error: old_str found {count} times, must be unique for safety."
                )
            new_content = old_content.replace(old_str, new_str, 1)

        file_path.write_text(new_content)
        logging.info(f"Successfully edited file {path}")
        return "OK"
    except Exception as e:
        logging.error(f"Failed to edit file {path}: {e}")
        return f"Error: An unexpected error occurred: {e}"


__all__ = ["bash", "edit_file", "list_files", "read_file"]
