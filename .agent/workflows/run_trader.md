---
description: Run the HolonicTrader Bot (Live or Paper)
---

This workflow guides you through starting the HolonicTrader bot.

1. **Activate Virtual Environment**
   Ensure you are using the correct virtual environment (`venv313`).
   ```powershell
   & .\.venv313\Scripts\Activate.ps1
   ```

2. **Check Configuration**
   Verify the trading mode in `config.py`.
   - `PAPER_TRADING = True` for Simulation.
   - `PAPER_TRADING = False` for Live Money.
   ```powershell
   Get-Content config.py | Select-String "PAPER_TRADING"
   ```

3. **Run the Bot**
   Start the main trading execution loop.
   ```powershell
   // turbo
   python main_live_phase4.py
   ```

4. **Monitor Logs**
   Check the latest log file to verify the system started correctly.
   ```powershell
   Get-ChildItem live_trading_session_*.log | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Tail 50
   ```
