import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

class DatabaseManager:
    """
    Manages SQLite connection and persistence for HolonicTrader.
    """
    def __init__(self, db_path: str = "holonic_trader.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Portfolio Table
        c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance_usd REAL,
            balance_asset REAL,
            updated_at TEXT
        )
        ''')
        
        # Migration: Add multi-asset columns if they don't exist
        try:
            c.execute("ALTER TABLE portfolio ADD COLUMN held_assets TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        try:
            c.execute("ALTER TABLE portfolio ADD COLUMN position_metadata TEXT")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        try:
            c.execute("ALTER TABLE portfolio ADD COLUMN fortress_balance REAL")
        except sqlite3.OperationalError:
            pass # Column likely exists
            
        try:
            c.execute("ALTER TABLE portfolio ADD COLUMN repayment_reserve REAL")
            print("[Database] Added repayment_reserve column to portfolio table")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                pass  # Column already exists
            else:
                raise  # Re-raise unexpected errors
        
        # Ledger Table (The Blockchain)
        c.execute('''
        CREATE TABLE IF NOT EXISTS ledger (
            hash TEXT PRIMARY KEY,
            prev_hash TEXT,
            timestamp TEXT,
            entropy_score REAL,
            regime TEXT,
            action TEXT
        )
        ''')
        
        # Trades Table (Performance Tracking)
        c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            quantity REAL,
            price REAL,
            cost_usd REAL,
            timestamp TEXT,
            pnl REAL,
            pnl_percent REAL
        )
        ''')
        
        # Migration: Add unrealized PnL columns if they don't exist
        try:
            c.execute("ALTER TABLE trades ADD COLUMN unrealized_pnl REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        try:
            c.execute("ALTER TABLE trades ADD COLUMN unrealized_pnl_percent REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
            
        # EXPERIMENT B: Metric Columns
        try:
            c.execute("ALTER TABLE trades ADD COLUMN mfe REAL")
            c.execute("ALTER TABLE trades ADD COLUMN mae REAL")
        except sqlite3.OperationalError:
            pass

        # 2026-03-21: Trade metadata columns for winning pattern analysis
        for col_name, col_type in [
            ('exit_reason', 'TEXT'),       # TARGET, STOP_LOSS, TRAILING, HARD_STOP, FORCED, HYGIENE
            ('strategy_type', 'TEXT'),     # WHALE_SHADOW, TREND, KALMAN_VALUE, etc.
            ('entropy_score', 'REAL'),     # Shannon entropy at entry (0-2.3)
            ('regime', 'TEXT'),            # ORDERED, TRANSITION, CHAOTIC
            ('conviction', 'REAL'),        # Oracle conviction at entry (0-1.0)
            ('quality_score', 'REAL'),     # Atlas quality score (0-100)
            ('is_whitelisted', 'INTEGER'), # 1 if whitelisted at entry time
        ]:
            try:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass
        
        # RL Experience Table (DQN Memory)
        c.execute('''
        CREATE TABLE IF NOT EXISTS rl_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            state TEXT,          -- JSON list
            action_idx INTEGER,
            reward REAL,
            next_state TEXT,     -- JSON list
            done BOOLEAN
        )
        ''')

        # Experience Memory Table (The Hippocampus)
        c.execute('''
        CREATE TABLE IF NOT EXISTS memory_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            context_vector TEXT, -- JSON list [RSI, BB, Vol, ...]
            outcome TEXT,        -- 'WIN' or 'LOSS'
            pnl_percent REAL,
            embedding_signature TEXT -- Optional: For faster indexing
        )
        ''')
        
        # SMCE State Table (Layer 0 Capital Doctrine)
        c.execute('''
        CREATE TABLE IF NOT EXISTS smce_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            smce_tier TEXT,
            day_start_equity REAL,
            week_start_equity REAL,
            last_day_reset TEXT,
            last_week_reset INTEGER,
            defensive_cooldown_until REAL,
            risk_multiplier_smce REAL,
            consecutive_clean_days INTEGER,
            period_max_drawdown REAL,
            allocation_pct_boost REAL,
            updated_at TEXT
        )
        ''')
        
        # Migration: Add allocation_pct_boost if missing
        try:
            c.execute("ALTER TABLE smce_state ADD COLUMN allocation_pct_boost REAL")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        conn.close()

    def save_trade(self, trade_data: Dict[str, Any]):
        """Save a executed trade to the DB."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO trades (symbol, direction, quantity, price, cost_usd, timestamp, pnl, pnl_percent, unrealized_pnl, unrealized_pnl_percent, mfe, mae, exit_reason, strategy_type, entropy_score, regime, conviction, quality_score, is_whitelisted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['symbol'],
            trade_data['direction'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data['cost_usd'],
            trade_data['timestamp'],
            trade_data.get('pnl', 0.0),
            trade_data.get('pnl_percent', 0.0),
            trade_data.get('unrealized_pnl', 0.0),
            trade_data.get('unrealized_pnl_percent', 0.0),
            trade_data.get('mfe', 0.0),
            trade_data.get('mae', 0.0),
            trade_data.get('exit_reason'),
            trade_data.get('strategy_type'),
            trade_data.get('entropy_score'),
            trade_data.get('regime'),
            trade_data.get('conviction'),
            trade_data.get('quality_score'),
            trade_data.get('is_whitelisted'),
        ))
        conn.commit()
        conn.close()

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent trades."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        
        return [{'symbol': r[1], 'direction': r[2], 'quantity': r[3], 'price': r[4], 
                 'cost_usd': r[5], 'timestamp': r[6], 'pnl': r[7], 'pnl_percent': r[8]} 
                for r in rows]
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent trades for win rate calculation.
        Returns only completed trades with PnL data.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT symbol, direction, pnl, pnl_percent, timestamp 
            FROM trades 
            WHERE pnl IS NOT NULL AND pnl != 0
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        
        return [{'symbol': r[0], 'direction': r[1], 'pnl': r[2], 
                 'pnl_percent': r[3], 'timestamp': r[4]} 
                for r in rows]

    def get_win_rate(self, limit: int = 50) -> float:
        """
        Calculate current win rate for PPO reward signal.
        Returns: float 0.0 to 1.0
        """
        try:
            trades = self.get_recent_trades(limit)
            if not trades:
                return 0.5 # Neutral baseline
                
            wins = len([t for t in trades if t['pnl'] > 0])
            total = len(trades)
            
            return wins / total if total > 0 else 0.5
        except Exception as e:
            print(f"⚠️ DB Error calculating win rate: {e}")
            return 0.5

    def save_portfolio(self, usd: float, held_assets: Dict[str, float], position_metadata: Dict[str, Any], fortress_balance: float = 0.0):
        """Save or update the portfolio state with explicit column mapping."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        assets_json = json.dumps(held_assets)
        meta_json = json.dumps(position_metadata)
        
        # Explicit column names in INSERT to match the actual table schema
        c.execute('''
        INSERT OR REPLACE INTO portfolio (id, balance_usd, held_assets, position_metadata, fortress_balance, repayment_reserve, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?)
        ''', (usd, assets_json, meta_json, fortress_balance, getattr(self, '_last_repayment_reserve', 0.0), timestamp))
        
        conn.commit()
        conn.close()

    def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Retrieve the last saved portfolio state using row_factory for safety."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute('SELECT balance_usd, held_assets, position_metadata, fortress_balance, repayment_reserve FROM portfolio WHERE id = 1')
        row = c.fetchone()
        conn.close()
        
        if row:
            # Deserialize
            held_assets = json.loads(row['held_assets']) if row['held_assets'] else {}
            position_metadata = json.loads(row['position_metadata']) if row['position_metadata'] else {}

            # Handle missing columns gracefully (old database schema)
            fortress_balance = row['fortress_balance'] if 'fortress_balance' in row.keys() else 0.0
            repayment_reserve = row['repayment_reserve'] if 'repayment_reserve' in row.keys() else 0.0

            return {
                'balance_usd': row['balance_usd'],
                'held_assets': held_assets,
                'position_metadata': position_metadata,
                'fortress_balance': fortress_balance or 0.0,
                'repayment_reserve': repayment_reserve or 0.0
            }
        return None

    def add_block(self, block_data: Dict[str, Any]):
        """Save a ledger block to the DB."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT OR IGNORE INTO ledger (hash, prev_hash, timestamp, entropy_score, regime, action)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            block_data['hash'],
            block_data['prev_hash'],
            block_data['timestamp'],
            block_data['entropy_score'],
            block_data['regime'],
            block_data['action']
        ))
        
        conn.commit()
        conn.close()

    def get_last_block(self) -> Optional[Dict[str, Any]]:
        """Get the most recent block added (by timestamp for simplicity)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Ordering by timestamp desc is decent proxy if clocks are stable
        c.execute('SELECT * FROM ledger ORDER BY timestamp DESC LIMIT 1')
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                'hash': row[0],
                'prev_hash': row[1],
                'timestamp': row[2],
                'entropy_score': row[3],
                'regime': row[4],
                'action': row[5]
            }
        return None
    def save_experience(self, experience: Dict[str, Any]):
        """Save an RL transition tuple to the DB."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO rl_experiences (timestamp, symbol, state, action_idx, reward, next_state, done)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            experience.get('timestamp', datetime.now().isoformat()),
            experience.get('symbol', 'UNKNOWN'),
            json.dumps(experience['state']),
            experience['action_idx'],
            experience['reward'],
            json.dumps(experience['next_state']),
            experience['done']
        ))
        
        conn.commit()
        conn.close()

    def get_experiences(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """Retrieve recent RL experiences for memory replay."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # We want the MOST RECENT experiences, but we might want to return them in chronological order
        # for sequential validity if using LSTM, but for DQN random sample it handles it.
        # However, memory.append adds to right. So we should fetch DESC (newest) but append them...
        # If we fetch 2000 newest, we should probably append them. 
        # Actually random sampling doesn't care about order.
        c.execute('SELECT * FROM rl_experiences ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            try:
                results.append({
                    'state': json.loads(row['state']),
                    'action_idx': int(row['action_idx']),
                    'reward': float(row['reward']),
                    'next_state': json.loads(row['next_state']),
                    'done': bool(row['done'])
                })
            except Exception:
                continue
                
        return results

    def save_smce_state(self, state_data: Dict[str, Any]):
        """Save SMCE L0 Capital Doctrine state."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        c.execute('''
        INSERT OR REPLACE INTO smce_state (
            id, smce_tier, day_start_equity, week_start_equity,
            last_day_reset, last_week_reset, defensive_cooldown_until,
            risk_multiplier_smce, consecutive_clean_days, period_max_drawdown,
            allocation_pct_boost, updated_at
        )
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state_data.get('smce_tier', 'SMALL'),
            state_data.get('day_start_equity', 0.0),
            state_data.get('week_start_equity', 0.0),
            str(state_data.get('last_day_reset', datetime.utcnow().date().isoformat())),
            int(state_data.get('last_week_reset', datetime.utcnow().isocalendar()[1])),
            float(state_data.get('defensive_cooldown_until', 0.0)),
            float(state_data.get('risk_multiplier_smce', 1.0)),
            int(state_data.get('consecutive_clean_days', 0)),
            float(state_data.get('period_max_drawdown', 0.0)),
            float(state_data.get('allocation_pct_boost', 0.0)),
            timestamp
        ))
        conn.commit()
        conn.close()

    def load_smce_state(self) -> Optional[Dict[str, Any]]:
        """Load SMCE L0 Capital Doctrine state."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM smce_state WHERE id = 1')
        row = c.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None

    def save_repayment_reserve(self, reserve: float):
        """Standalone update for repayment reserve."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        self._last_repayment_reserve = reserve 
        c.execute("UPDATE portfolio SET repayment_reserve = ?, updated_at = ? WHERE id = 1", 
                  (reserve, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def load_repayment_reserve(self) -> float:
        """Standalone load for repayment reserve."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("SELECT repayment_reserve FROM portfolio WHERE id = 1")
            row = c.fetchone()
            conn.close()
            if row and row[0] is not None:
                return float(row[0])
        except sqlite3.OperationalError:
            pass  # Column doesn't exist yet
        return 0.0
