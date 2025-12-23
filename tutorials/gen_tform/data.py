import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def download_if_needed(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    print(f"Downloading dataset to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")

def load_text(path: str, url: str = DATA_URL) -> str:
    download_if_needed(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
