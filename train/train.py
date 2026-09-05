"""
Same tanh char-RNN the ESP32 runs, trained with Adam + minibatches in PyTorch.

    h = tanh(Wxh[:, x] + Whh h + bh)
    z = Why h + by
    P = softmax(z)

AdaGrad-in-a-Python-loop was too slow at H=956. Adam on batched
sequences is the same model, fewer steps, fatter matmuls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = "abcdefghijklmnopqrstuvwxyz 0123456789.,:?!'-\n"
HIDDEN_SIZE = 956
SEQ_LENGTH = 48
BATCH = 64
EPOCHS = 32
LR = 1.5e-3
DEMO_PROMPT = "you: what does the rocket do"


def filter_text(raw: str, vocab: str) -> str:
    allowed = set(vocab)
    text = raw.lower().replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch if ch in allowed else " " for ch in text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip() + "\n"


class CharRNN(nn.Module):
    def __init__(self, v: int, h: int):
        super().__init__()
        self.Wxh = nn.Parameter(torch.randn(h, v) * 0.01)
        self.Whh = nn.Parameter(torch.randn(h, h) * 0.01)
        self.Why = nn.Parameter(torch.randn(v, h) * 0.01)
        self.bh = nn.Parameter(torch.zeros(h))
        self.why_bias = nn.Parameter(torch.zeros(v))

    def step(self, ix: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ix (B,), h (B, H)  — column pick, same as the chip
        h = torch.tanh(self.Wxh[:, ix].T + h @ self.Whh.T + self.bh)
        z = h @ self.Why.T + self.why_bias
        return z, h


def prep_you_prompt(prompt: str) -> str:
    if prompt.lower().lstrip().startswith("you:") and not prompt.endswith("\n"):
        return prompt + "\n"
    return prompt


@torch.no_grad()
def continue_from(model: CharRNN, prompt: str, n: int, stoi: dict, itos: dict, device, temp=0.6) -> str:
    model.eval()
    prompt = prep_you_prompt(prompt)
    h = torch.zeros(1, model.Whh.shape[0], device=device)
    z = None
    for ch in prompt:
        if ch not in stoi:
            continue
        z, h = model.step(torch.tensor([stoi[ch]], device=device), h)
    if z is None:
        z, h = model.step(torch.tensor([stoi.get(" ", 0)], device=device), h)
    out = prompt
    v = model.Why.shape[0]
    for _ in range(n):
        p = torch.softmax(z[0] / max(temp, 1e-6), dim=0).cpu().numpy()
        ix = int(np.random.choice(v, p=p))
        out += itos[ix]
        z, h = model.step(torch.tensor([ix], device=device), h)
    model.train()
    return out


def count_params(model: CharRNN) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "chat_corpus.txt")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "checkpoints" / "rnn.npz")
    parser.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--seq", type=int, default=SEQ_LENGTH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = VOCAB
    v = len(vocab)
    h = args.hidden
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    if not args.corpus.exists():
        raise SystemExit(f"Missing {args.corpus}. Run build_chat_corpus.py first.")

    data = filter_text(args.corpus.read_text(encoding="utf-8"), vocab)
    ids = np.array([stoi[c] for c in data], dtype=np.int64)
    if len(ids) < args.seq + 2:
        raise SystemExit(f"Corpus too short ({len(data)} chars).")

    n_windows = (len(ids) - 1) // args.seq
    x_all = ids[: n_windows * args.seq].reshape(n_windows, args.seq)
    y_all = ids[1 : n_windows * args.seq + 1].reshape(n_windows, args.seq)

    model = CharRNN(v, h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device   : {device}")
    print(f"corpus   : {len(data)} characters  windows={n_windows}")
    print(f"vocab    : {v}")
    print(f"hidden   : {h}")
    print(f"seq/batch: {args.seq}/{args.batch}")
    print(f"params   : {count_params(model)}")
    print(f"opt      : Adam lr={args.lr}  epochs={args.epochs}")

    step = 0
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n_windows)
        total, ntok = 0.0, 0
        for s in range(0, n_windows, args.batch):
            idx = perm[s : s + args.batch]
            if len(idx) < 2:
                continue
            xb = torch.tensor(x_all[idx], device=device)
            yb = torch.tensor(y_all[idx], device=device)
            bsz = xb.size(0)
            hidden = torch.zeros(bsz, h, device=device)
            loss = 0.0
            for t in range(args.seq):
                z, hidden = model.step(xb[:, t], hidden)
                loss = loss + F.cross_entropy(z, yb[:, t])
            loss = loss / args.seq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * bsz * args.seq
            ntok += bsz * args.seq
            step += 1

        ppl = float(np.exp(min(total / max(ntok, 1), 20)))
        demo = continue_from(model, DEMO_PROMPT, 70, stoi, itos, device)
        print(f"\n[{epoch:02d}/{args.epochs}] loss={total / max(ntok, 1):.3f}  ppl={ppl:.1f}")
        print(f"  sample: {demo!r}")

    def dump(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy().astype(np.float32).copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        Wxh=dump(model.Wxh),
        Whh=dump(model.Whh),
        Why=dump(model.Why),
        bh=dump(model.bh),
        by=dump(model.why_bias),
        vocab=np.array(vocab),
        hidden_size=np.array(h),
    )
    np.savez(args.out, **payload)
    print(f"\nsaved weights -> {args.out}")
    print("torch:", continue_from(model, "you: what is the f-1", 80, stoi, itos, device, temp=0.4))

    from generate import generate

    check = generate("you: what is the f-1", 80, args.out, 0.4)
    print("npz:  ", check)
    if "bot:" not in check.lower() or "kerolox" not in check.lower():
        raise SystemExit("npz verify failed — disk weights do not match the live model")


if __name__ == "__main__":
    main()
