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
        
        conn.commit()
        conn.close()

    def save_trade(self, trade_data: Dict[str, Any]):
        """Save a executed trade to the DB."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
        INSERT INTO trades (symbol, direction, quantity, price, cost_usd, timestamp, pnl, pnl_percent, unrealized_pnl, unrealized_pnl_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            trade_data.get('unrealized_pnl_percent', 0.0)
        ))
        conn.commit()
        conn.close()

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent trades."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT * FROM trades ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def save_portfolio(self, usd: float, held_assets: Dict[str, float], position_metadata: Dict[str, Any]):
        """Save or update the portfolio state with explicit column mapping."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        assets_json = json.dumps(held_assets)
        meta_json = json.dumps(position_metadata)
        
        # Explicit column names in INSERT to match the actual table schema
        c.execute('''
        INSERT OR REPLACE INTO portfolio (id, balance_usd, held_assets, position_metadata, updated_at)
        VALUES (1, ?, ?, ?, ?)
        ''', (usd, assets_json, meta_json, timestamp))
        
        conn.commit()
        conn.close()

    def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Retrieve the last saved portfolio state using row_factory for safety."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('SELECT balance_usd, held_assets, position_metadata FROM portfolio WHERE id = 1')
        row = c.fetchone()
        conn.close()
        
        if row:
            # Deserialize
            held_assets = json.loads(row['held_assets']) if row['held_assets'] else {}
            position_metadata = json.loads(row['position_metadata']) if row['position_metadata'] else {}
            
            return {
                'balance_usd': row['balance_usd'], 
                'held_assets': held_assets,
                'position_metadata': position_metadata
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
