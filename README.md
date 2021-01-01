# Promixal Policy Optimization with PyTorch  
This repository implements [promixal policy optimization](https://arxiv.org/abs/1707.06347) using the PyTorch Lightning package. PyTorch Lightning helps reduce boilerplate code and modularize model training. Hence, different parts such as the loss function, advantage calculation, or training configurations can be easily modified as per users' experiments. 

This implementation is inspired by OpenAI baselines for [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2) and implementation 
of other RL algorithms in [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts/)

## Details 
This PPO implemenation works with both discrete and continous action-space environments via OpenAI Gym. Implements PPO Actor-Critic Style. Continous actor uses normal distribution to predict actions wheres discrete actor uses multinomial distribution (Categorical distribution in Torch) to predict actions. 


## Requirements 
* Python3 >= 3.6 
* OpenAI Gym 
* PyTorch
* PyTorch Lightning 

## Results 
TBA
