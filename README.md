# PokeAgent 🎮🤖

A python-based intelligent Pokémon Showdown Bot driven by **Gemini 3.1 Pro Preview** and the `poke-env` environment. Built to play Gen 9 Random Battles natively on a local server.

## Overview

PokeAgent utilizes Google's state-of-the-art **Gemini 3.1 Pro Preview** model to analyze battle states, select optimal moves, and switch Pokémon. It interfaces directly with Pokémon Showdown's protocol using [`poke-env`](https://poke-env.readthedocs.io/), providing full control over the battle and access to complete game state data.

### Key Features
- **Intelligent Decision Engine**: Gemini parses the board position, active status effects, available moves, and health values to act optimally.
- **Local Testing Environment**: Bundled with a local Pokémon Showdown server setup for continuous testing against baseline bots without needing to connect to the public official server.
- **Poke-Env Integration**: Clean abstractions for Pokémon Showdown battles through the `poke-env` Player classes.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js (for compiling and running the showdown server)
- A Google API Key (`GEMINI_API_KEY`)

### Installation & Setup

1. **Clone the Directory**
   ```bash
   git clone https://github.com/AlbertQiSun/PokeAgent.git
   cd PokeAgent
   ```

2. **Run Setup Script**
   An automated setup script is provided to initialize submodules, install showdown's Node.js dependencies, and build a Python virtual environment automatically:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Set Up the API Key**
   The `setup.sh` script generates a `.env` file for you automatically. Open `.env` at the root of the project and paste in your Gemini API Key safely.

### Running the Bot

First, start your local Pokémon Showdown instance. You can use the provided script (make sure `pokemon-showdown` is set up via `./setup.sh`!):
```bash
./start_server.sh
```

In a separate terminal, launch your bots natively using:
```bash
./start.sh
# OR
python bot/main.py
```
This script initializes the local showdown connection, logs in both your Gemini-powered driver and a Random baseline antagonist, and tests them asynchronously.

### Architecture Highlights
- `bot/main.py`: Sets up the `LocalhostServerConfiguration` and coordinates the battle between the bots. 
- `bot/gemini_player.py`: Contains the `GeminiPlayer` override, prompting the `gemini-3.1-pro-preview` model for structured JSON outputs formatting move choices.

---
🚀 *Happy Battling!*
