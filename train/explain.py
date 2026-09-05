"""
Show the forward-pass math after training is done.

Not TinyML. Not TFLite. Same RNN we trained:

    h = tanh(Wxh x + Whh h + bh)
    z = Why h + by
    P = softmax(z / T)

For each next character: scores, exp, probabilities, and why the winner won.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def show_char(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "' '"
    return ch


def explain_step(vocab: str, z_raw: np.ndarray, p: np.ndarray, picked: int, context: str) -> None:
    order = np.argsort(-p.ravel())
    print(f"  after {context!r}")
    print(f"  {'char':<6} {'z':>8} {'P':>8}")
    for k in order[:5]:
        mark = "  <-- picked" if k == picked else ""
        print(f"  {show_char(vocab[k]):<6} {z_raw.ravel()[k]:+8.3f} {p.ravel()[k]:8.1%}{mark}")
    print()


def run(prompt: str, n: int, npz: Path, temperature: float) -> str:
    from generate import load_vocab, prep_you_prompt

    data = np.load(npz, allow_pickle=True)
    Wxh, Whh, Why = data["Wxh"], data["Whh"], data["Why"]
    bh = data["bh"].reshape(-1, 1)
    by = data["by"].reshape(-1, 1)
    vocab = load_vocab(data)
    prompt = prep_you_prompt(prompt)
    if n <= 0:
        n = 16
    stoi = {ch: i for i, ch in enumerate(vocab)}
    H, V = Wxh.shape

    print("from-scratch char-RNN  (no TinyML, no TFLite)")
    print(f"H={H}  V={V}  T={temperature}")
    print("h = tanh(Wxh x + Whh h + bh)")
    print("z = Why h + by")
    print("P = softmax(z / T)")
    print()

    h = np.zeros((H, 1))
    for ch in prompt.lower():
        if ch not in stoi:
            continue
        x = np.zeros((V, 1))
        x[stoi[ch]] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)

    out = list(prompt.lower())
    context = prompt.lower()
    for _ in range(n):
        z_raw = Why @ h + by
        z = z_raw / max(temperature, 1e-6)
        p = softmax(z)
        ix = int(np.random.choice(V, p=p.ravel()))
        explain_step(vocab, z_raw, p, ix, context)
        out.append(vocab[ix])
        context += vocab[ix]
        if len(context) > 40:
            context = context[-40:]
        x = np.zeros((V, 1))
        x[ix] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)

    text = "".join(out)
    print("full:")
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=Path(__file__).parent / "checkpoints" / "rnn.npz")
    parser.add_argument("--prompt", default="you: what does the rocket do")
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--temp", type=float, default=0.6)
    args = parser.parse_args()
    if not args.npz.exists():
        raise SystemExit(f"Missing {args.npz}. Run train.py first.")
    run(args.prompt, args.n, args.npz, args.temp)


if __name__ == "__main__":
    main()
