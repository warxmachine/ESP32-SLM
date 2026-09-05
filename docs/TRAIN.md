# Train on a PC

The ESP32 never trains. It only runs the forward pass.

## Setup

```powershell
cd train
python -m pip install -r requirements.txt
```

`numpy` and `torch` are required to train. `pyserial` is required only for `chat.py`.

## Rocket Q&A corpus (1M model)

Facts live in `build_chat_corpus.py` (`HAND_QA`). Rebuild a repeated training file:

```powershell
python build_chat_corpus.py --skip-kaggle
```

That writes `chat_corpus.txt` (gitignored). Optional: `--with-kaggle` if you drop launch CSVs in `train/data/` (or `~/Downloads/archive`). CSV rows drowned the facts last time; hand-only is the shipped recipe.

Keep text inside the vocab:

```text
abcdefghijklmnopqrstuvwxyz 0123456789.,:?!'-\n
```

## Fit and export

```powershell
python train.py
python export.py
```

- Default: **H=956**, **V=45**, **~1,000,977** parameters, Adam, 32 epochs.
- `train.py` writes `checkpoints/rnn.npz` (gitignored) and checks the file with `generate.py` before exiting.
- `export.py` writes `../firmware/02_TinyLM/model_weights.h` (int8 + per-matrix scale).

Then flash with **Huge APP** ([FLASH.md](FLASH.md)).

## PC inference (same math as the chip)

```powershell
python generate.py --prompt "you: what is the f-1"
python explain.py --prompt "you: what is the f-1" -n 12 --temp 0.6
```

`-n 0` on `generate.py` means “until a blank line.” `you:` prompts get a trailing newline so they match training (`you: ...\\nbot: ...`). The chip does the same. PC samples use float32 `npz` weights; the board uses int8 + scale, so answers will be close, not bit-identical.

## Tiny 3,006-param toy (video 1)

That run used `corpus.txt`, a smaller vocab (no digits), `HIDDEN_SIZE=32`, and a NumPy AdaGrad loop that is **no longer in this repo**. `train.py` is Adam at H=956. Recreating the toy means writing a small trainer (or shrinking `--hidden 32`, pointing `--corpus` at `corpus.txt`, and accepting that the optimizer is still Adam). The firmware here is the **1M** export unless you change those defaults and export again.

## Extra scripts

| Script | Role |
| --- | --- |
| `chat.py` | Serial chat (close Arduino Serial Monitor first). Needs `pyserial`. `--why` turns the chip's why print **on**. |
| `export.py` | float32 `npz` → int8 C header |
| `explain.py` | Top-5 `z` / `P` per letter on the PC |
