"""PC-side inference — same equations the ESP32 will run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    s = float(np.sum(e))
    if s < 1e-8:
        s = 1e-8
    return e / s


def load_vocab(data) -> str:
    raw = data["vocab"]
    try:
        return str(raw.item())
    except (AttributeError, ValueError):
        return str(raw)


def prep_you_prompt(prompt: str) -> str:
    """Training pairs are `you: ...\\nbot: ...`. Feed that newline or the RNN stays on the question line."""
    if prompt.lower().lstrip().startswith("you:") and not prompt.endswith("\n"):
        return prompt + "\n"
    return prompt


def generate(prompt: str, n: int, npz: Path, temperature: float) -> str:
    data = np.load(npz, allow_pickle=True)
    Wxh, Whh, Why = data["Wxh"], data["Whh"], data["Why"]
    bh = data["bh"].reshape(-1, 1)
    by = data["by"].reshape(-1, 1)
    vocab = load_vocab(data)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    prompt = prep_you_prompt(prompt)

    H, V = Wxh.shape
    h = np.zeros((H, 1))
    fed = False
    for ch in prompt.lower():
        if ch not in stoi:
            continue
        x = np.zeros((V, 1))
        x[stoi[ch]] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        fed = True
    if not fed:
        x = np.zeros((V, 1))
        x[stoi.get(" ", 0)] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)

    out = list(prompt.lower())
    unlimited = n <= 0
    blanks = 0
    made = 0
    while unlimited or made < n:
        z = (Why @ h + by) / max(temperature, 1e-6)
        p = softmax(z)
        ix = int(np.random.choice(V, p=p.ravel()))
        ch = vocab[ix]
        out.append(ch)
        made += 1
        if ch == "\n":
            blanks += 1
            if unlimited and blanks >= 2:
                break
        else:
            blanks = 0
        x = np.zeros((V, 1))
        x[ix] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path(__file__).parent / "checkpoints" / "rnn.npz")
    parser.add_argument("--prompt", default="you: what does the rocket do")
    parser.add_argument("-n", type=int, default=0, help="new letters; 0 = until a blank line")
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--why", action="store_true", help="print softmax math (use explain.py)")
    args = parser.parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing {args.npz}. Run train.py first.")
    if args.why:
        from explain import run
        why_n = args.n if args.n > 0 else 16
        run(args.prompt, why_n, args.npz, args.temp)
        return
    print(generate(args.prompt, args.n, args.npz, args.temp), end="")
    print()


if __name__ == "__main__":
    main()
