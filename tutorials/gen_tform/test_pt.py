import os
import torch

from tokeniser import decode
from utils_pt import CharTransformer

def main():
    ckpt_path = os.path.join("ckpt_pt.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    chars = ckpt["chars"]
    config = ckpt["config"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = CharTransformer(**config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = "ROMEO:\n"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=500, temperature=0.9)[0].tolist()
    print(decode(out, chars))

if __name__ == "__main__":
    main()
