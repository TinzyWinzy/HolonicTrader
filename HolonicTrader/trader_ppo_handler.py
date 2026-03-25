"""
trader_ppo_handler.py
PPO Reward Calculation Logic - Extracted from agent_trader.py

Handles:
- Asymmetric loss aversion
- Time decay / velocity rewards
- Mission psychology (Operation Centurion)
"""
import numpy as np
import pandas as pd
import config


def calculate_ppo_reward(pnl_res: float, meta: dict, current_equity: float, 
                         holon_name: str = "TraderNexus") -> tuple[float, float]:
    """
    Calculate shaped PPO reward with time decay and mission psychology.
    
    Returns:
        tuple: (reward, duration_mins)
    """
    if meta.get('ppo_state') is None or meta.get('ppo_conviction') is None:
        return 0.0, 0.0
    
    pnl_pct = pnl_res  # pnl_res is percentage
    is_win = pnl_pct > 0
    
    # 1. Base Utility - Scale small % (0.01) to recognizable scalar (0.1)
    reward = pnl_pct * 10.0
    
    # 2. Asymmetric Punishment (Loss Aversion)
    if not is_win:
        reward *= 2.5  # Pain Factor
        
    # 3. Time Duration
    entry_ts_iso = meta.get('entry_timestamp')
    duration_mins = 1.0  # Default minimum
    if entry_ts_iso:
        try:
            t_entry = pd.to_datetime(entry_ts_iso)
            t_exit = pd.Timestamp.now(tz='UTC')
            duration_mins = (t_exit - t_entry).total_seconds() / 60.0
        except:
            pass
            
    duration_mins = max(1.0, duration_mins)
    
    # 4. Time Decay / Velocity
    # log1p(duration) -> log(2) ~ 0.69, log(61) ~ 4.1
    time_factor = np.log1p(duration_mins)
    if time_factor < 1.0:
        time_factor = 1.0  # Floor at 1
    
    if is_win:
        # STRATEGY ALIGNMENT: Scalp-to-Pyramid
        # If short duration (< 60m), reward velocity (Scalp).
        # If long duration (> 60m), do NOT penalize (Pyramid/Trend).
        if duration_mins < 60:
            reward = reward / time_factor  # Fast wins > Slow scalps
        # else: keep reward as-is for trend following
    else:
        reward = reward * time_factor  # Long losses > Fast losses (Double Pain)

    # === MISSION PSYCHOLOGY (Operation Centurion) ===
    initial_capital = getattr(config, 'INITIAL_CAPITAL', 100.0)
    mission_target = getattr(config, 'MISSION_TARGET', 1000.0)
    
    # 1. Calculate Progress
    mission_progress = (current_equity - initial_capital) / (mission_target - initial_capital)
    mission_progress = max(0.0, min(1.0, mission_progress))  # Clamp 0-1

    # 2. Progress Booster (Mid-Game Motivation)
    if reward > 0:
        reward *= (1.0 + mission_progress)

    # 3. Proximity Defense (End-Game Anxiety)
    if reward < 0 and mission_progress > 0.80:
        print(f"[{holon_name}] 🛡️ PROXIMITY DEFENSE ACTIVE: Double Penalty applied.")
        reward *= 2.0
        
    return reward, duration_mins


def store_ppo_experience(ppo, meta: dict, reward: float, holon_name: str = "TraderNexus"):
    """Store experience in PPO memory."""
    if ppo is None or meta is None:
        return
        
    if meta.get('ppo_state') is None or meta.get('ppo_conviction') is None:
        return
    
    pnl_pct = reward / 10.0  # Approximate reverse for logging
    duration_mins = 1.0  # Will be recalculated or passed
    
    state = np.array(meta['ppo_state'])
    ppo.remember(state, meta['ppo_conviction'], reward, 0.0, 0.0, True)
