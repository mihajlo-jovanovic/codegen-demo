import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import list_files, read_file


def test_read_file_returns_file_contents(tmp_path):
    file_path = tmp_path / "example.txt"
    content = "hello world"
    file_path.write_text(content)

    result = read_file.invoke({"path": str(file_path)})

    assert result == content


def test_list_files_ignores_hidden_directories(tmp_path, monkeypatch):
    root_dir = tmp_path / "project"
    hidden_dir = root_dir / ".hidden"
    visible_dir = root_dir / "visible"
    hidden_dir.mkdir(parents=True)
    visible_dir.mkdir(parents=True)

    (hidden_dir / "secret.txt").write_text("secret")
    (visible_dir / "public.txt").write_text("public")

    monkeypatch.chdir(root_dir)

    result = list_files.invoke({})
    files = json.loads(result)

    assert "visible/" in files
    assert "visible/public.txt" in files
    assert not any(path.startswith(".hidden") for path in files)
