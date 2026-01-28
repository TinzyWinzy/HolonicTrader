# Holonic Trader - Docker Setup

This project has been dockerized to run the Backend (Python/Flask + Rust Engine) and Frontend (React/Vite) in isolated containers.

## Prerequisites
- Docker Desktop installed and running.

## Services
1. **Backend**: Runs the trading bot and API server.
   - Port: `5000`
   - Mounts: `logs/`, `holonic_trader.db`, `user_config.json`, `hall_of_fame.json` (Data is persisted on host)
2. **Frontend**: Serves the UI via Nginx.
   - Port: `8080` (Mapped to internal port 80)
   - Connects to Backend via proxy (`/api`, `/socket.io`).

## How to Run

1. **Start the System**:
   ```bash
   docker-compose up --build
   ```

2. **Access the Dashboard**:
   Open [http://localhost:8080](http://localhost:8080) in your browser.

3. **Stop the System**:
   Press `Ctrl+C` in the terminal or run:
   ```bash
   docker-compose down
   ```

## Notes
- The initial build may take a few minutes as it compiles the Rust engine and installs Python dependencies.
- If you change the frontend code, you need to rebuild: `docker-compose up --build`.
- Ensure `holonic_trader.db` exists in the root directory before starting, or it will be created as a directory by Docker (if it doesn't exist). A placeholder file works best.
