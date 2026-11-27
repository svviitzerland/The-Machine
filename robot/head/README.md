# Head

The Machine's first body: an ESP32 with two servos that turns to look at whoever it is talking to. The laptop stays the brain; the head joins your WiFi and The Machine talks to it there.

## Try it in the simulator

1. Open this folder (`robot/head`) in VS Code with the [Wokwi extension](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode).
2. Build: `pio run -e sim`
3. Press F1 and run **Wokwi: Start Simulator**. Keep the simulator tab visible, or it pauses.
4. Start The Machine with the simulated head:

```bash
python -m the_machine run --body localhost:7351
```

Move in front of your webcam and the servos follow you.

## The real head

Parts: an ESP32 DevKitC V4 (or any ESP32 dev board), two SG90 servos, a pan-tilt bracket and a USB webcam.

| Servo | Signal | Power |
|---|---|---|
| pan (left and right) | GPIO 18 | 5V, GND |
| tilt (up and down) | GPIO 19 | 5V, GND |

Copy `src/secrets.example.h` to `src/secrets.h`, fill in your WiFi, then flash it:

```bash
pio run -e head -t upload
python -m the_machine run --body machine-head.local:7351
```

After the first flash over USB, later updates can go over WiFi: `pio run -e head-ota -t upload`.

## Protocol

One line per command, over TCP port 7351 or USB serial at 115200 baud. Every command is answered with `OK P<pan> T<tilt>`.

| Command | Meaning |
|---|---|
| `P90 T60` | look at pan 90°, tilt 60° (either part can be left out) |
| `S300` | turn at 300° per second |
| `?` | report where it is looking |
