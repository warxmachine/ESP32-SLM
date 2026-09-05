# Contributing

This is a small, readable stack. Keep it that way.

- The chip runs **inference only**. Do not add training, Wi-Fi, or a tokenizer to the `.ino` unless that is the point of the change.
- `firmware/02_TinyLM/model_weights.h` is **generated**. Change training + `export.py`, do not hand-edit the arrays. Commit a new header only when it is the intended shipped checkpoint.
- Do not commit `train/checkpoints/`, `train/chat_corpus.txt`, Kaggle dumps, or `kaggle.json`.
- Prompts in docs should match `HAND_QA` in `train/build_chat_corpus.py` if you want them to work on the shipped weights.
- Say what the model is not: it is not an LLM.

Flash notes stay in [docs/FLASH.md](docs/FLASH.md).
