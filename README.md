# PokeEnv RL Project
Project for the Reinforcement Learning course (2025-1)

# Installation

## Prerequisites
Make sure you have [Python](https://www.python.org/downloads/) 3.11 or higher installed on your system. You can check your Python version by running:
```bash
python --version
```
Also, you will need to have [Docker](https://www.docker.com/) installed to run the Pokemon Showdown environment. You can check if Docker is installed by running:
```bash
docker --version
```

> Alternatively, you can use [Node.js](https://nodejs.org/en/download/) to run the Pokemon Showdown environment. Follow the instructions in the [PokeEnv documentation](https://poke-env.readthedocs.io/en/stable/getting_started.html#configuring-a-pokemon-showdown-server) to set it up.

## Clone the Repository
To get started, clone this repository and install the required packages using pip (we recommend using a virtual environment):

```bash
pip install -r requirements.txt
```

# Usage
To run the project, you can use the following command:
```bash
python main.py [OPTIONS]
```
This will start a CLI to choose the desired environment and run the training process. To see the available options, you can run:
```bash
python main.py --help
```

# About
This repository contains a reinforcement learning project based on the work of Hamish Ivison in his [stunfisk-rl Repository](https://github.com/hamishivi/stunfisk-rl). The project uses the environment from the [PokeEnv](https://poke-env.readthedocs.io/en/stable/index.html) with the [Pokemon Showdown](https://pokemonshowdown.com/) battle simulator.
