# Traffic Signal RL Environment

## Description
This environment simulates traffic signal control to reduce jams.

## Action Space
- 0: Red
- 1: Green
- 2: Orange

## Observation Space
- Queue length of cars at intersection.

## Setup
```bash
pip install torch gym numpy pyyaml
python inference.py