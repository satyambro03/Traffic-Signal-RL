---
title: Traffic Signal RL
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
sdk_version: "1.0"
python_version: "3.9"
app_file: inference.py
pinned: false
---

# Traffic Signal RL Environment 🚦

## Overview
This repository contains reinforcement learning environments designed for the Meta PyTorch OpenEnv Hackathon.  
The agent learns to optimize tasks such as email sorting and traffic signal control using the standard `step()` / `reset()` API.

## Tasks
### 1. Email Sorting (Easy)
- **Action Space**: Discrete(3) → [Work, Personal, Spam]  
- **Observation Space**: 10-dimensional vector  
- **Reward**: 1.0 for correct classification, 0.0 otherwise  

### 2. Traffic Signal Control (Medium)
- **Action Space**: Discrete(3) → [Red, Green, Orange]  
- **Observation Space**: Queue length of cars at a single intersection  
- **Reward**: Positive for reduced waiting time, 0.0 if jam persists  

### 3. Multi-Intersection Optimization (Hard)
- **Action Space**: Discrete(6) → control multiple signals simultaneously  
- **Observation Space**: Cars per lane + traffic density  
- **Reward**: 1.0 for smooth flow during rush hour, partial reward for improvement  

## Reward System
- **EmailSort** → Reward = 1.0 if email sorted correctly, else 0.0  
- **TrafficSignal** → Reward = 1.0 if chosen signal reduces queue length, else 0.0  
- **MultiIntersection** → Reward = 1.0 for optimal flow, partial rewards (0.2–0.5) for improvements, else 0.0  

## Files
- `env.py` → Environment definitions (EmailSortEnv, TrafficSignalEnv, MultiIntersectionEnv)  
- `openenv.yaml` → Task specifications for all 3 tasks  
- `inference.py` → Baseline agent (random actions, reproducible scores)  
- `Dockerfile` → Deployment setup (Gymnasium + NumPy + PyYAML + Gradio)  
- `README.md` → Documentation and hackathon notes  

## Setup
```bash
pip install gymnasium numpy pyyaml gradio
python inference.py

## Deployment
This project is deployed on Hugging Face Spaces using Docker.  
👉 [Live Demo Here](https://huggingface.co/spaces/Satyam-Vishwakarma/Traffic-Signal-RL)