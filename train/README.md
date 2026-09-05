# Training and PC tools

See [docs/TRAIN.md](../docs/TRAIN.md) for the full pipeline. Python **3.9+**.

| File | Purpose |
| --- | --- |
| `build_chat_corpus.py` | Build `you:` / `bot:` text from hand facts (optional `--with-kaggle`) |
| `train.py` | Adam + batched BPTT (PyTorch), saves `checkpoints/rnn.npz` |
| `export.py` | Quantize to int8, write `firmware/02_TinyLM/model_weights.h` |
| `generate.py` | PC sample (same equations as the chip) |
| `explain.py` | Print top-5 `z` and `P` per letter |
| `chat.py` | Serial I/O with the board (`pyserial`) |
| `corpus.txt` | Original tiny rocket sentences (3k-param era) |

`checkpoints/rnn.npz` and `chat_corpus.txt` are gitignored. Clone-and-flash uses the header already in `firmware/`; PC generate needs a train run first.

```text
python -m pip install -r requirements.txt
python build_chat_corpus.py
python train.py
python export.py
python generate.py --prompt "you: what is the f-1"
```
