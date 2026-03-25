import time
import random
import json
import os
import logging
import numpy as np
from evolution_lab import EvolutionLab
from HolonicTrader.validation_gate import LiveValidationGate
from HolonicTrader.evolution_monitor import EvolutionMonitor

# Force UTF-8 encoding at module level (Windows fix)
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Config
LOOP_INTERVAL = 60
GENOME_FILE = 'live_genome.json'
WINNER_ARCHIVE = 'winner_challenge.json'
STAGNATION_THRESHOLD = 3
MAX_STAGNATION = 10

# Import Config
import config
if not hasattr(config, 'ALLOWED_ASSETS'):
    config.ALLOWED_ASSETS = list(config.KRAKEN_SYMBOL_MAP.keys())

import sys
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# --- BLOOMBERG TERMINAL THEME ---
bloomberg_theme = Theme({
    "logging.level.info": "cyan",           # Info is Cyan
    "logging.level.warning": "bold yellow", # Warnings scream
    "logging.level.error": "bold white on red", # Critical
    "repr.number": "bold gold3",            # Numbers are Gold
    "repr.str": "white",                    # Strings are White
})

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

console = Console(theme=bloomberg_theme, force_terminal=False)

# 1. File Handler (Text / CSV-like)
file_handler = logging.FileHandler('evo_engine.log', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s')
file_handler.setFormatter(file_formatter)

# 2. Rich Console Handler (Bloomberg Style)
# Rich handles timestamps and levels natively, so we just pass the message.
rich_handler = RichHandler(
    console=console, 
    rich_tracebacks=True,
    markup=True,
    show_time=True,
    show_path=False
)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

# 3. Setup Root Logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, rich_handler]
)

class Island:
    """A distinct population with unique evolutionary rules."""
    def __init__(self, name: str, strategy_bias: str, mutation_rate: float, min_leverage: float, allowed_categories: list):
        self.name = name
        self.strategy_bias = strategy_bias # 'AGGRESSIVE', 'BALANCED', 'CONSERVATIVE'
        self.mutation_rate = mutation_rate
        self.average_leverage = min_leverage
        self.allowed_categories = allowed_categories
        self.champion = None
        self.stagnation = 0

class ArchipelagoEngine:
    def __init__(self):
        self.total_cycles = 0
        self.migration_interval = 3
        
        # 1. Initialize Islands (Parallel Evolution & Category Specialization)
        self.islands = [
            Island("🌋 Volcano Peak", "AGGRESSIVE", 0.35, 1.0, ['meme_coin', 'alt_coin']),
            Island("🏰 Iron Fort", "CONSERVATIVE", 0.05, 1.0, ['large_cap']),
            Island("🏝️ Darwin's Rock", "BALANCED", 0.15, 1.0, ['mid_cap', 'large_cap'])
        ]
        
        # Shared Knowledge
        self.hall_of_fame = self.load_hall_of_fame()
        self.best_global_fitness = max([g.get('fitness', 0) for g in self.hall_of_fame]) if self.hall_of_fame else 0.0
        logging.info(f"🏆 HOF Loaded: {len(self.hall_of_fame)} Kings. Best Fitness: {self.best_global_fitness:.2f}")
        
        # Labs (One per Island concept - logic handled in loop)
        self.lab = EvolutionLab(
            symbols=config.ALLOWED_ASSETS,
            initial_capital=config.INITIAL_CAPITAL,
            leverage=20.0, # Base leverage, overridden by Island
            history_limit=1500,
            secondary_limit=6000
        )
        
        # 2. Gate & Monitor (Tasks 5 & 6)
        self.gate = LiveValidationGate(paper_period_hours=48)
        self.monitor = EvolutionMonitor()
        
        # 3. Dashboard Communication
        self.status_file = 'evolution_status.json'
        self.last_validation = {}  # Track last validation result
        self.fitness_history = {island.name: [] for island in self.islands}
        self.recent_alerts = []  # Last 20 alerts
        
    def _write_status_file(self):
        """Write evolution status for dashboard to read."""
        try:
            # Get global champion from HOF
            global_champ = None
            if self.hall_of_fame:
                global_champ = max(self.hall_of_fame, key=lambda x: x.get('fitness', 0))
            
            status = {
                'last_update': time.time(),
                'total_cycles': self.total_cycles,
                'best_global_fitness': self.best_global_fitness,
                'global_champion': {
                    'fitness': global_champ.get('fitness', 0) if global_champ else 0,
                    'genome': global_champ.get('genome', {}) if global_champ else {},
                    'source_island': global_champ.get('source_island', 'N/A') if global_champ else 'N/A'
                } if global_champ else None,
                'islands': [
                    {
                        'name': island.name,
                        'strategy_bias': island.strategy_bias,
                        'fitness': island.champion.get('fitness', 0) if island.champion else 0,
                        'stagnation': island.stagnation,
                        'categories': island.allowed_categories,
                        'history': self.fitness_history.get(island.name, [])[-20:]  # Last 20 points
                    }
                    for island in self.islands
                ],
                'validation_gate': self.last_validation,
                'alerts': self.recent_alerts[-20:]  # Last 20 alerts
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logging.error(f"Status file write error: {e}")
        
    def load_seed(self):
        # ... logic to load seed ...
        try:
             if os.path.exists(GENOME_FILE):
                with open(GENOME_FILE, 'r') as f: return json.load(f).get('genome')
        except: pass
        return None

    def load_hall_of_fame(self):
        try:
            if os.path.exists('hall_of_fame.json'):
                with open('hall_of_fame.json', 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list): return data
        except Exception as e:
            logging.error(f"Failed to load HOF: {e}")
        return []

    def save_apex(self, result, source_island):
        # VALIDATION: Prevent Fitness Collapse
        current_fit = result.get('fitness', 0)
        if self.best_global_fitness > 1000 and current_fit < self.best_global_fitness * 0.5:
            logging.warning(f"🛑 REJECTING APEX SAVE: Fitness Collapse Detected ({current_fit:.2f} < {self.best_global_fitness:.2f})")
            return

        # --- BRAIN STABILITY GATE (Tuning Phase) ---
        # Prevent rapid whipsaws by rejecting massive fitness jumps (Variance > 5.0) which indicate overfitting
        if self.best_global_fitness > 0:
            variance = abs(current_fit - self.best_global_fitness)
            if variance > 5.0:
                 logging.warning(f"🛑 STABILITY GATE: Rejecting unstable fitness jump (Var {variance:.2f} > 5.0). Fit: {current_fit:.2f}")
                 return
        # -------------------------------------------

        result['timestamp'] = time.time()
        result['source_island'] = source_island
        
        # Save Active Genome
        with open(GENOME_FILE, 'w') as f:
            json.dump(result, f, indent=4)
            
        # Save Hall of Fame (Top 10)
        # Deduplicate and Sort
        # Deduplicate and Sort
        # Merge active HOF with potentially loaded one (in case another process updated it? No, single process)
        # Update memory HOF
        
        unique_hof = {json.dumps(g['genome'], sort_keys=True): g for g in self.hall_of_fame}.values()
        sorted_hof = sorted(unique_hof, key=lambda x: x['fitness'], reverse=True)[:10]
        
        with open('hall_of_fame.json', 'w') as f:
            json.dump(list(sorted_hof), f, indent=4)
            
        logging.info(f"💾 GLOBAL APEX SAVED from {source_island} (Fit: {result.get('fitness', 0):.2f})")
        logging.info(f"💾 HOF UPDATED ({len(sorted_hof)} Kings)")

    def run(self):
        logging.info("🏝️ ARCHIPELAGO ENGINE STARTING...")
        
        # Seed Initial Populations
        seed = self.load_seed()
        self.lab.seed_population(seed)
        
        while True:
            try:
                self.total_cycles += 1
                logging.info(f"=== Archipelago Cycle {self.total_cycles} ===")
                
                # 1. Rotate through Islands
                for island in self.islands:
                    logging.info(f"⛵ Sailing to {island.name}...")
                    
                    # A. Adjust Lab Environment for this Island
                    # (In a real parallel system, these would serve simultaneous processes)
                    self.lab.leverage = island.average_leverage
                    
                    try:
                        # B. Portfolio Baskets (Robustness Test)
                        # FIX #5 2026-03-02: INCREASED basket size 2 -> 4 to reduce stagnation
                        # FIX 2026-03-20: Pin basket per island per cycle for consistent fitness comparison
                        # Problem: Random basket each generation = noisy fitness comparison across genomes
                        # Solution: Fix basket for entire island run, rotate across cycles
                        basket_size = 4

                        # Ensure we pick valid symbols that match the ISLAND CATEGORY
                        island_syms = []
                        for s in self.lab.symbols:
                            if s not in self.lab.datasets: continue
                            
                            try:
                                cat = self.lab.profiler.classify_category(s)
                            except:
                                cat = 'unknown'
                                
                            if cat in island.allowed_categories:
                                island_syms.append(s)

                        if not island_syms:
                            logging.warning(f"⚠️ {island.name} has NO valid symbols for categories {island.allowed_categories}. Skipping.")
                            continue
                        
                        # Deterministic rotation: cycle through island_syms in fixed windows
                        # Each cycle rotates by basket_size so all assets get tested over time
                        island_syms.sort()  # Stable ordering
                        rotation_offset = (self.total_cycles * basket_size) % max(len(island_syms), 1)
                        rotated = island_syms[rotation_offset:] + island_syms[:rotation_offset]
                        basket = rotated[:basket_size]
                        
                        # C. Evolve
                        winner = self.lab.evolve(
                            generations=3, 
                            mutation_rate=island.mutation_rate,
                            basket=basket
                        )
                        
                        # D. Assess & Monitor
                        current_fitness = winner.get('fitness', 0)
                        
                        # Run Health Check (Task 6)
                        alerts = self.monitor.check_health(island.name, winner)
                        for alert in alerts:
                            logging.warning(alert)
                            self.recent_alerts.append({'time': time.time(), 'msg': alert})
                        
                        # Track fitness history
                        self.fitness_history[island.name].append(current_fitness)
                        
                        if current_fitness > self.best_global_fitness:
                            # E. VALIDATION GATE (Task 5)
                            # We pick the first asset in the basket as the primary test subject
                            primary_asset = basket[0] if basket else config.BTC_SYMBOL
                            
                            # INJECTION: Pass pre-loaded data to avoid disk read errors
                            # FIX: Prevent Data Leakage - Only pass the VALIDATION portion (last 30%)
                            primary_dataset = self.lab.datasets.get(primary_asset)
                            primary_df = None
                            
                            if primary_dataset and primary_dataset.df is not None:
                                full_len = len(primary_dataset.df)
                                split_idx = int(full_len * 0.7)
                                if full_len > 100:
                                    primary_df = primary_dataset.df.iloc[split_idx:]
                            
                            is_promoted, reason, result = self.gate.validate_genome(winner['genome'], primary_asset, external_df=primary_df)
                            
                            # Store validation result for dashboard
                            self.last_validation = {
                                'timestamp': time.time(),
                                'symbol': primary_asset,
                                'status': 'PROMOTED' if is_promoted else 'REJECTED',
                                'reason': reason,
                                'stats': result.get('stats', {}) if result else {}
                            }
                            
                            if is_promoted:
                                logging.info(f"🚀 {island.name} produced a NEW GLOBAL CHAMPION! (Fit {current_fitness:.2f}) - PROMOTED")
                                self.best_global_fitness = current_fitness
                                island.champion = winner
                                self.hall_of_fame.append(winner)
                                self.save_apex(winner, island.name)
                            else:
                                logging.warning(f"🚫 {island.name} Champion REJECTED by Gate: {reason}")
                        else:
                            # Local Improvement check?
                            pass
                            
                    except Exception as island_error:
                         logging.error(f"⚠️ Error Processing Island {island.name}: {island_error}")
                         continue
                        
                # 2. Migration Phase (Mixing)
                if self.total_cycles % self.migration_interval == 0:
                    logging.info("🕊️ MIGRATION EVENT: Sharing best genomes between islands.")
                    # In this sequential implementation, self.lab already has the mixed population.
                    # We just need to make sure we inject HOF members back into the gene pool
                    # if they were lost.
                    if self.hall_of_fame:
                         # Reinject Top 3 to prevent Drift forgetting
                         hof_top = sorted(self.hall_of_fame, key=lambda x: x['fitness'], reverse=True)[:3]
                         for elite in hof_top:
                             if elite['genome'] not in self.lab.population:
                                 self.lab.population.append(elite['genome'])
                         logging.info(f"💉 Re-injected {len(hof_top)} Ancient Kings from Hall of Fame.")

                # 3. Write status for dashboard
                self._write_status_file()
                
                logging.info(f"Sleeping {LOOP_INTERVAL}s...")
                time.sleep(LOOP_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Archipelago Error: {e}", exc_info=True)
                time.sleep(60)

def main():
    engine = ArchipelagoEngine()
    engine.run()

if __name__ == "__main__":
    main()