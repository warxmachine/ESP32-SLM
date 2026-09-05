# ESP32 Tiny Language Model

A **from-scratch character-level RNN**. Train on a PC. Run **inference only** on an ESP32-WROOM-32.

No cloud. No API. No TensorFlow. No tokenizer. Not ChatGPT.

```
character → one-hot (pick a column of Wxh)
         → h = tanh(Wxh x + Whh h + bh)
         → z = Why h + by
         → P = softmax(z / T)
         → next character
```

The GitHub name says SLM. The model is a **1M-parameter char-RNN**, not a transformer.

It grew in three steps: a **3,006**-parameter toy that continues `the rock`, a look **inside softmax** (`z` and `P`), then the **1,000,977**-parameter rocket Q&A model shipped here. That last model still predicts **one letter at a time**. The 3k trainer is not in this tree.

## What it is / is not

| It is | It is not |
| --- | --- |
| A vanilla tanh RNN you can read in one `.ino` | A transformer or LLM |
| Next-character sampling (temperature 0.6 by default) | A chatbot that “understands” questions |
| int8 weights in flash + one scale per tensor | Training on the microcontroller |
| Exact string gravity: `you: what is the f-1` works | `what is f1` meaning the same thing |

Almost all of the 1M weights are **hidden → hidden**: `956 × 956 = 913,936` connections.

## Current checkpoint (what you flash)

| | |
| --- | --- |
| Parameters | **1,000,977** (`Wxh` 43,020 + `Whh` 913,936 + `Why` 43,020 + biases) |
| Hidden / vocab | **H = 956**, **V = 45** (`a–z`, digits, `.,:?!'-`, space, newline) |
| Train | PC, PyTorch **Adam**, batched BPTT |
| On device | Forward pass only, int8 + scale |
| Weight file | [`firmware/02_TinyLM/model_weights.h`](firmware/02_TinyLM/model_weights.h) (~4 MB of C arrays, ~1 MB in flash) |

## Repo layout

```
docs/                  Flash, train, and demo prompts
firmware/
  01_BlinkTest/        Serial hello (no LED) — prove the board
  02_TinyLM/           RNN inference + generated weights header
train/                 Corpus, train, export, PC generate
CONTRIBUTING.md        How to change this without turning it into an LLM
LICENSE                MIT
```

## Quick start — flash the rocket model

You do **not** need Python if `model_weights.h` is already in the tree.

Hardware: ESP32-WROOM-32 DevKit, **4 MB flash**, USB **data** cable. No PSRAM required.

1. Install [Arduino IDE](https://www.arduino.cc/en/software) 2.x and the **esp32** core (Espressif).  
   Board URL: `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. **Tools → Board → ESP32 Dev Module**
3. **Tools → Partition Scheme → Huge APP (3MB No OTA / 1MB SPIFFS)** — recommended. The sketch is ~1.28 MB; the default app slot is only ~1.31 MB.
4. **Tools → Port → COMx**
5. Open [`firmware/02_TinyLM/02_TinyLM.ino`](firmware/02_TinyLM/02_TinyLM.ino) and Upload.
6. Serial Monitor **115200**. Close any other program using the port first.

After reset, wait for `params=1000977  hidden=956  vocab=45`. The chip then generates `you: what does the rocket do` on its own. That is the boot demo, not a hang.

If upload fails: hold **BOOT**, tap **EN**, release **BOOT**, upload again.

Full steps: [docs/FLASH.md](docs/FLASH.md).

Type **exactly** (lowercase, include `you:`, hyphen in `f-1`):

```text
you: what is the f-1
you: what is a nozzle
you: how does a rocket fly
you: are you chatgpt
```

More lines that match training: [docs/PROMPTS.md](docs/PROMPTS.md).

On the chip: `/why` toggles top-5 `z` / `P`. `/temp 0.2` is stabler. `/n 0` is unlimited (stops on a blank line).

## Quick start — PC (no board)

```text
cd train
python -m pip install -r requirements.txt
python generate.py --prompt "you: what is the f-1"
```

Needs Python **3.9+** and `train/checkpoints/rnn.npz` (gitignored, not in the clone). Train first if it is missing — [docs/TRAIN.md](docs/TRAIN.md). Clone-and-flash does **not** need this file.

## Train your own, then export

```text
cd train
python build_chat_corpus.py
python train.py
python export.py
```

Default corpus is the hand-written facts in `build_chat_corpus.py`. That is what the shipped weights were trained on — not Kaggle or other third-party dumps.

`export.py` overwrites `firmware/02_TinyLM/model_weights.h`. Then flash with **Huge APP**.

The 3,006-parameter toy used [`train/corpus.txt`](train/corpus.txt) and `HIDDEN_SIZE=32`. Recreating it takes extra work (`docs/TRAIN.md`). The shipped firmware is the 1M rocket model.

## License

[MIT](LICENSE). The `.ino`, training scripts, and the generated int8 header in this repo are included. Retrain if you want a different voice; do not hand-edit `model_weights.h`.
