# Flash the ESP32

## Hardware

- ESP32-WROOM-32 DevKit (4 MB flash). No PSRAM needed.
- USB **data** cable (charge-only cables never get a COM port).
- Windows: vendor USB-UART driver (this project was used with a Silicon Labs **CP210x** on COM3).

## Arduino IDE

1. **File → Preferences → Additional Board Manager URLs**

   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`

2. **Tools → Board → Boards Manager** → install **esp32 by Espressif Systems**.
3. **Tools → Board → ESP32 Arduino → ESP32 Dev Module**.
4. **Tools → Partition Scheme → Huge APP (3MB No OTA / 1MB SPIFFS)**.  
   Default app size is ~1.3 MB. This sketch is ~1.28 MB of **model + runtime**. Huge APP is the safe choice.
5. **Tools → Port → COMx** (the port that appears when you plug in).
6. Open `firmware/02_TinyLM/02_TinyLM.ino` (it includes `model_weights.h`).
7. Upload.
8. **Serial Monitor, 115200 baud**. Only one program can own the port: close Monitor before `python train/chat.py`.

Stuck in bootloader: hold **BOOT** (GPIO0), tap **EN**, release **BOOT**, upload again.

## Arduino CLI

```text
arduino-cli compile --fqbn esp32:esp32:esp32:PartitionScheme=huge_app firmware/02_TinyLM
arduino-cli upload  --fqbn esp32:esp32:esp32:PartitionScheme=huge_app -p COM3 firmware/02_TinyLM
```

Change `COM3` to your port.

## Prove the USB stack first (optional)

Flash `firmware/01_BlinkTest/01_BlinkTest.ino` with the **default** partition if you want. Serial at 115200 should print `ESP32 AI CORE ONLINE` once a second. That sketch is not the language model.

## After reset

Wait for:

```text
params=1000977  hidden=956  vocab=45
```

Boot runs `you: what does the rocket do`. Then type a trained prompt and press Enter.

`/why` — print top-5 logits `z` and softmax `P` after each character (off by default).  
`/temp 0.2` — less random.  
`/n 80` — hard cap. `/n 0` — no cap, stop on a blank line.
