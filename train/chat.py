"""Interactive serial chat with the ESP32 TinyLM.

Close Arduino Serial Monitor first (only one program can own COM3).

Opening the port resets the board. Firmware boots with n unlimited and why off.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import serial
from serial.tools import list_ports


def find_port(preferred: str | None) -> str:
    if preferred:
        return preferred
    for p in list_ports.comports():
        desc = f"{p.device} {p.description} {p.hwid}"
        if "10C4" in desc or "CP210" in desc or "Silicon Labs" in desc:
            return p.device
    ports = list(list_ports.comports())
    if len(ports) == 1:
        return ports[0].device
    names = ", ".join(p.device for p in ports) or "none"
    raise SystemExit(f"Could not find the ESP32. Ports: {names}\nPass --port COM3")


def wait_boot(ser: serial.Serial, timeout: float = 90.0) -> None:
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
            if b"Type a prompt" in buf:
                return
    if not buf:
        print("(no boot text — board may still be generating)")


def reader(ser: serial.Serial, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            n = ser.in_waiting
            data = ser.read(n or 1)
        except serial.SerialException:
            break
        if data:
            sys.stdout.write(data.decode("utf-8", errors="replace"))
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None, help="e.g. COM3")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("-n", type=int, default=None, help="override letters per generate")
    parser.add_argument("--why", action="store_true", help="turn why on after boot (firmware starts with why off)")
    parser.add_argument("--no-why", action="store_true", help="leave why off (this is already the boot default)")
    args = parser.parse_args()

    port = find_port(args.port)
    ser = serial.Serial(port, args.baud, timeout=0.2)
    print(f"Connected to {port}. Waiting for boot story to finish...")
    wait_boot(ser)

    stop = threading.Event()
    t = threading.Thread(target=reader, args=(ser, stop), daemon=True)
    t.start()

    time.sleep(0.2)
    if args.n is not None:
        ser.write(f"/n {args.n}\n".encode("utf-8"))
        time.sleep(0.15)
    if args.why:
        ser.write(b"/why\n")
        time.sleep(0.15)
    if args.temp is not None:
        ser.write(f"/temp {args.temp}\n".encode("utf-8"))
        time.sleep(0.15)

    try:
        while True:
            line = input("YOU> ").strip()
            if not line:
                continue
            ser.write((line + "\n").encode("utf-8"))
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    finally:
        stop.set()
        ser.close()


if __name__ == "__main__":
    main()
