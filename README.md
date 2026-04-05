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
This repository contains reinforcement learning environments designed for the hackathon.  
The agent learns to optimize tasks such as email sorting and traffic signal control.

## Tasks
1. **Email Sorting (Easy)**  
   - Action Space: Discrete(3) → [Work, Personal, Spam]  
   - Observation Space: 10-dimensional vector  
   - Reward: 1.0 for correct classification, 0.0 otherwise  

2. **Traffic Signal Control (Medium)**  
   - Action Space: Discrete(3) → [Red, Green, Orange]  
   - Observation Space: Queue length of cars at a single intersection  
   - Reward: Positive for reduced waiting time, 0.0 if jam persists  

3. **Multi-Intersection Optimization (Hard)**  
   - Action Space: Control multiple signals simultaneously  
   - Observation Space: Cars per lane + time of day  
   - Reward: 1.0 for smooth flow during rush hour, partial reward for improvement  

## Files
- `env.py` → Environment definitions  
- `openenv.yaml` → Task specifications  
- `inference.py` → Baseline agent (random actions)  
- `Dockerfile` → Deployment setup  

## Setup
```bash
pip install torch gym numpy pyyaml
python inference.py