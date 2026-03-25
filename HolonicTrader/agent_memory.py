"""
MemoryHolon - The Hippocampus of HolonicTrader
Phase 47: Episodic Experience Replay

Goal: Store trade contexts and outcomes to provide "Déjà Vu" signals to the Oracle.
Mechanism: k-Nearest Neighbors (cosine similarity) on context vectors.
"""

from HolonicTrader.holon_core import Holon, Disposition
import numpy as np
import json
import math
from typing import List, Dict, Tuple, Any

class MemoryHolon(Holon):
    def __init__(self, name="MemoryHippocampus", db_manager=None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.8))
        self.db_manager = db_manager
        self.vector_cache = [] # List of {'vector': np.array, 'outcome': str, 'pnl': float}
        self.last_sync_time = 0
        
        # Load initially
        if self.db_manager:
            self._load_vectors()
            
    def _load_vectors(self):
        """Load all memory vectors from DB into RAM for fast search."""
        try:
            conn = self._get_conn()
            c = conn.cursor()
            c.execute("SELECT context_vector, outcome, pnl_percent FROM memory_vectors ORDER BY id DESC LIMIT 1000")
            rows = c.fetchall()
            conn.close()
            
            self.vector_cache = []
            for r in rows:
                try:
                    vec = np.array(json.loads(r[0]))
                    outcome = r[1]
                    pnl = r[2]
                    self.vector_cache.append({'vector': vec, 'outcome': outcome, 'pnl': pnl})
                except:
                    continue
            
            print(f"[{self.name}] 🧠 Loaded {len(self.vector_cache)} experiences.")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Load failed: {e}")

    def _get_conn(self):
        import sqlite3
        return sqlite3.connect(self.db_manager.db_path)

    def store_experience(self, context_vector: List[float], outcome: str, pnl_percent: float, symbol: str):
        """Save a new experience to DB and Memory."""
        try:
            timestamp = "NOW" # Logic handled by DB default or caller? DB Manager usually handles this.
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            # 1. DB Persist
            conn = self._get_conn()
            c = conn.cursor()
            vec_json = json.dumps(context_vector)
            c.execute("INSERT INTO memory_vectors (timestamp, symbol, context_vector, outcome, pnl_percent) VALUES (?, ?, ?, ?, ?)",
                      (timestamp, symbol, vec_json, outcome, pnl_percent))
            conn.commit()
            conn.close()
            
            # 2. RAM Update
            self.vector_cache.append({
                'vector': np.array(context_vector),
                'outcome': outcome,
                'pnl': pnl_percent
            })
            
            print(f"[{self.name}] 💾 Defined new memory: {symbol} -> {outcome} ({pnl_percent*100:.2f}%)")
            
        except Exception as e:
            print(f"[{self.name}] ❌ Store failed: {e}")

    def query_memory(self, current_vector: List[float], k: int = 5) -> Dict[str, Any]:
        """
        Recall: Find k-nearest past experiences.
        Returns: {'deja_vu_score': float (-1.0 to 1.0), 'details': str}
        """
        if not self.vector_cache:
            return {'deja_vu_score': 0.0, 'details': "No Memory"}
            
        target = np.array(current_vector)
        target_norm = np.linalg.norm(target)
        if target_norm == 0: return {'deja_vu_score': 0.0, 'details': "Zero Vector"}
        
        # Calculate Cosine Similarity for all
        scored_memories = []
        for mem in self.vector_cache:
            vec = mem['vector']
            norm = np.linalg.norm(vec)
            if norm == 0: continue
            
            sim = np.dot(target, vec) / (target_norm * norm)
            scored_memories.append((sim, mem))
            
        # Sort by similarity desc
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_memories[:k]
        
        # Calculate Déjà Vu Score
        # Weighted average of PnL outcomes, weighted by similarity
        # If very similar setup led to loss, score is negative.
        weighted_pnl = 0.0
        total_weight = 0.0
        
        log_hits = []
        
        for sim, mem in top_k:
            if sim < 0.8: continue # Ignore loose matches
            
            # Weight: Similarity^2 (emphasize exact matches)
            weight = sim * sim
            pnl_sign = 1.0 if mem['pnl'] > 0 else -1.0
            
            # Magnitude matters? Yes, +10% is better than +1%
            # Clamp PnL influence to [-1, 1] range roughly
            impact = max(-1.0, min(1.0, mem['pnl'] * 10)) # 10% move = 1.0 impact
            
            weighted_pnl += (impact * weight)
            total_weight += weight
            
            log_hits.append(f"{mem['outcome']}({sim:.2f})")
            
        if total_weight == 0:
            return {'deja_vu_score': 0.0, 'details': "No Match"}
            
        final_score = weighted_pnl / total_weight
        
        # Normalize to [-1, 1]
        final_score = max(-1.0, min(1.0, final_score))
        
        details = f"Recall({len(log_hits)}): " + ", ".join(log_hits[:3])
        return {'deja_vu_score': final_score, 'details': details}

    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages for health checks or data requests."""
        if content == "HEALTH_CHECK":
            return {
                "status": "OK",
                "cached_experiences": len(self.vector_cache),
                "db_connected": self.db_manager is not None
            }
        elif content == "RELOAD":
            self._load_vectors()
            return {"status": "RELOADED", "count": len(self.vector_cache)}
        return None
