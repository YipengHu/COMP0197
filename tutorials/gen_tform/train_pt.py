import os
import time
import torch

from data import load_text
from tokeniser import build_char_vocab, encode
from utils_pt import get_batch, CharTransformer

def main():
    # Files
    data_path = os.path.join("data", "tinyshakespeare.txt")
    ckpt_path = os.path.join("ckpt_pt.pt")

    # Hyperparameters (small demo)
    block_size = 128
    batch_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.1
    lr = 3e-4

    max_steps = 1200
    eval_every = 200
    seed = 42

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text = load_text(data_path)
    vocab = build_char_vocab(text)
    ids = torch.tensor(encode(text, vocab.stoi), dtype=torch.long)

    # Split
    n = int(0.9 * ids.numel())
    train_data = ids[:n]
    val_data = ids[n:]

    config = dict(
        vocab_size=vocab.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )

    model = CharTransformer(**config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def estimate_loss():
        model.eval()
        out = {}
        for name, d in [("train", train_data), ("val", val_data)]:
            losses = []
            for _ in range(20):
                xb, yb = get_batch(d, block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    t0 = time.time()
    model.train()
    for step in range(1, max_steps + 1):
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0:
            losses = estimate_loss()
            print(f"step {step:4d}, train {losses['train']:.4f}, val {losses['val']:.4f}, elapsed {time.time()-t0:.1f}s")

    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": vocab.chars,
            "config": config,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
