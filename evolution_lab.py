import random
import copy
import json
import logging
import os
import math
import numpy as np
from typing import List, Dict, Optional
from sandbox.playground import Playground
from sandbox.strategies.evo_strategy import EvoStrategy
from HolonicTrader.data_guard import DataGuard
from HolonicTrader.asset_profiler import AssetProfiler
import config

# Logger configured by parent (run_evolution_loop.py)
logger = logging.getLogger("EvolutionLab")

class EvolutionLab:
    def __init__(self, 
                 symbols: List[str] = None, # Default to None, load from config if missing
                 initial_capital: float = None,
                 leverage: float = 5.0, # REDUCED: 20x -> 5x for Survival
                 population_size: int = 60,
                 fitness_mode: str = 'BALANCED', # 'BALANCED' (Safety) or 'AGGRESSIVE' (Growth)
                 verbose: bool = False):
        
        self.symbols = symbols if symbols else list(config.KRAKEN_SYMBOL_MAP.keys())
        self.initial_capital = initial_capital if initial_capital else config.INITIAL_CAPITAL
        self.leverage = leverage
        self.population_size = population_size
        self.fitness_mode = fitness_mode
        self.verbose = verbose
        
        self.datasets: Dict[str, Playground] = {}
        self.population: List[Dict] = []
        self.hall_of_fame: List[Dict] = [] # Memory: Top Genomes
        self.best_overall: Optional[Dict] = None
        self.guard = DataGuard()
        self.profiler = AssetProfiler()

    def load_data(self, history_limit: int = 1500, secondary_limit: int = 6000):
        """Pre-load data for all arenas."""
        logger.info(f"📡 Pre-loading Gauntlet Data for {self.symbols}...")
        for sym in self.symbols:
            p = Playground(symbol=sym, initial_capital=self.initial_capital, verbose=False, leverage=self.leverage)
            p.load_data(limit=history_limit)
            p.load_secondary_data(limit=secondary_limit)
            
            # Profile Asset
            if p.df is not None and not p.df.empty:
                self.profiler.calculate_profile(sym, p.df)
                
            self.datasets[sym] = p
        logger.info("✅ Data Loaded.")

    def generate_random_genome(self) -> Dict:
        """Spawn random genome suitable for benchmark survival."""
        return {
            'rsi_buy': random.uniform(20, 50),     # Buy Dip
            'rsi_sell': random.uniform(75, 95),    # Sell Rip
            'stop_loss': random.uniform(0.01, 0.05), # 1% - 5% (Safe for 1x)
            'take_profit': random.uniform(0.05, 0.20), # Swing targets
            'use_satellite': True,
            'sat_rvol': random.uniform(2.0, 5.0),
            'sat_bb_expand': random.uniform(0.05, 0.25),
            
            # --- INSTITUTIONAL GENES ---
            'leverage_cap': round(random.uniform(1.0, 5.0), 2), # RESTORED: 1-5x (Golden Ratio)
            'trailing_activation': random.uniform(0.1, 1.5), # RECALIBRATED: 10% - 150% gain
            'trailing_distance': random.uniform(0.01, 0.05)  # RECALIBRATED: 1% - 5% trail
        }

    def normalize_gene(self, key: str, value: float) -> float:
        """Apply biological limits to prevent parameter explosion."""
        if key == 'rsi_buy':
            return max(10.0, min(60.0, value))
        elif key == 'rsi_sell':
            return max(40.0, min(90.0, value))
        elif key == 'stop_loss':
            # Hard Limit: 0.5% to 10% (Prevent 0.0001% stops that trigger on noise)
            return max(0.005, min(0.10, value))
        elif key == 'take_profit':
            # Hard Limit: 1% to 50%
            return max(0.01, min(0.50, value))
        elif key == 'leverage_cap':
            return max(1.0, min(5.0, value))
        elif key == 'sat_rvol':
            # Soft Limit: Tanh compression for > 5.0
            if value > 5.0:
                return 5.0 + math.tanh(value - 5.0) * 5.0 # Maxes around 10
            return max(1.0, value)
        elif key == 'trailing_activation':
            # Hard Limit: 5% to 200% activation
            return max(0.05, min(2.0, value))
        elif key == 'trailing_distance':
            # Hard Limit: 0.5% to 10% distance
            return max(0.005, min(0.10, value))
        else:
            return value



    def seed_population(self, elite_genome: Optional[Dict] = None):
        """Initialize population, optionally injecting an Elite (e.g., current live brain)."""
        self.population = [self.generate_random_genome() for _ in range(self.population_size)]
        
        if elite_genome:
            # Validate elite keys
            valid = True
            for k in self.population[0].keys():
                if k not in elite_genome:
                    valid = False
            
            # Merge Elite with defaults to ensure all keys exist
            if valid:
                self.population[0] = copy.deepcopy(elite_genome)
                logger.info(f"🧬 Injected Elite Genome: {elite_genome}")
            else:
                # Patch missing keys instead of ignoring
                defaults = self.generate_random_genome()
                defaults.update(elite_genome) # Overlay elite values
                self.population[0] = defaults
                logger.info("🧬 Injected Elite Genome (Patched with new genes).")
        
        # SANITIZATION: Enforce Physics
        # If High Leverage, FORCE Tight Stops.
        # This prevents "Toxic Legacy Genes" (e.g. 6% SL) from bankrupting the simulation instantly.
        elite = self.population[0]
        if elite.get('leverage_cap', 1.0) > 20.0:
            if elite.get('stop_loss', 0.05) > 0.02: # 2% max for 50x
                logger.warning(f"⚠️ Sanitizing Toxic Stop Loss: {elite['stop_loss']} -> 0.015")
                elite['stop_loss'] = 0.015
                self.population[0] = elite

    def run_gauntlet(self, genome: Dict, basket: List[str] = None) -> Dict:
        """
        Run the Gauntlet: Calculates average performance across the asset basket.
        Logic: Portfolio-wide Reliability (Parallel Evaluation)
        """
        active_symbols = basket if basket else self.symbols
        
        # Aggregate Stats
        total_roi = 0.0
        total_sharpe = 0.0
        total_sortino = 0.0
        total_max_dd = 0.0
        total_trades = 0
        total_wins = 0
        
        valid_arenas = 0
        
        # Test each asset independently (Fixed Capital per Asset)
        test_capital = 1000.0 
        
        for symbol in active_symbols:
            if symbol not in self.datasets: continue
            
            arena = copy.copy(self.datasets[symbol])
            if arena.df is None or arena.df.empty: 
                continue
            
            # --- DATA FIX: Ensure minimum history for valid split ---
            # Need at least 500 bars for meaningful Training + 200 for Validation
            if len(arena.df) < 500: # Relaxed from 700 to allow newer coins
                # logger.warning(f"⚠️ {symbol} skipped: Insufficient Data ({len(arena.df)} rows < 500)")
                continue
            # ----------------------------------------------------
            
            # Split Data: 70% Train, 30% Val
            full_df = arena.df
            split_idx = int(len(full_df) * 0.7)
            
            # Index Logic Protection
            if split_idx < 100 or split_idx >= len(full_df) - 50:
                 continue

            # --- PHASE A: TRAIN ---
            arena.df = full_df.iloc[:split_idx]
            arena.capital = test_capital
            arena.inventory = 0.0
            arena.margin_locked = 0.0
            arena.trades = []
            arena.equity_curve = []
            
            # Inject Genome
            arena.leverage = genome.get('leverage_cap', 1.0)
            arena.inject_strategy(EvoStrategy(genome))
            arena.run()
            
            # Capture Metrics (Train)
            train_roi = (arena.capital - test_capital) / test_capital
            
            # --- PHASE B: VALIDATION (Out-of-Sample) ---
            arena.df = full_df.iloc[split_idx:]
            arena.capital = test_capital # Reset capital for validation
            arena.inventory = 0.0
            arena.margin_locked = 0.0
            arena.trades = []
            arena.equity_curve = []
            
            arena.run()
            val_roi = (arena.capital - test_capital) / test_capital
            
            # Restore full DF
            arena.df = full_df
            
            # Capture Metrics for this Asset (Focus on Validation for Fitness)
            if arena.trades:
                valid_arenas += 1
                
                # --- DATA GLITCH GUARD (Advanced) ---
                is_valid, reason = self.guard.validate_roi(symbol, val_roi)
                if not is_valid:
                    logger.warning(f"🛡️ DATA GUARD: Ignoring {symbol} Glitch ROI {val_roi*100:.1f}%. Reason: {reason}")
                    valid_arenas -= 1 # Revert count
                    continue # Skip adding this toxic asset to the average
                # -------------------------
                
                total_roi += val_roi
                
                # 2. Trades & Wins
                curr_trades = [t for t in arena.trades if t['type']=='SELL']
                total_trades += len(curr_trades)
                total_wins += len([t for t in curr_trades if t['pnl']>0])
                
                # 3. Drawdown (Local)
                peak = test_capital
                local_max_dd = 0.0
                for item in arena.equity_curve:
                    if item['equity'] > peak: peak = item['equity']
                    dd = (peak - item['equity']) / peak
                    if dd > local_max_dd: local_max_dd = dd
                total_max_dd += local_max_dd
                
                # 4. Sharpe/Sortino (Local)
                import numpy as np
                if len(arena.equity_curve) > 2:
                    eq_series = [e['equity'] for e in arena.equity_curve]
                    returns = np.diff(eq_series) / eq_series[:-1]
                    returns = returns[returns != 0]
                    
                    if len(returns) > 5:
                        avg_ret = np.mean(returns)
                        std_dev = np.std(returns) + 1e-9
                        
                        # Annualize (Approx 15m/1h candles)
                        s_sharpe = min(avg_ret / std_dev, 5.0) # Cap at 5.0
                        
                        total_sharpe += s_sharpe
                        
                        # Calmar Ratio (Return / Max DD)
                        calmar = val_roi / max(local_max_dd, 0.01)
                        total_sortino += min(calmar, 10.0) # Using sortino slot for Calmar
            
        # === AVERAGE METRICS ===
        if valid_arenas > 0:
            avg_roi = total_roi / valid_arenas
            avg_sharpe = total_sharpe / valid_arenas
            avg_sortino = total_sortino / valid_arenas
            avg_dd = total_max_dd / valid_arenas
            win_rate = total_wins / total_trades if total_trades > 0 else 0
        else:
            return {
                'genome': genome, 'fitness': 0.0, 'roi': 0.0, 'max_dd': 1.0, 
                'final_equity': self.initial_capital, 'sharpe':0,'sortino':0, 'trades':0
            }
            
        # === FITNESS FUNCTION (Multi-Objective) ===
        # Prioritize Risk-Adjusted Returns > Raw ROI
        
        # 1. Survival Factor (Heavy Penalty for DD)
        survival_score = 1.0
        if avg_dd > 0.25: survival_score -= (avg_dd - 0.25) * 2.0 # Relaxed from 0.15
        if avg_dd > 0.50: survival_score = 0.2 # Relaxed death zone from 0.30
        
        # 2. Activity Check (Avoid dead strategies)
        # Relaxed: Only punish if practically zero trades (< 0.5 per arena)
        # 2. Activity Check (Avoid dead strategies)
        # Relaxed: Only punish if practically zero trades (< 0.5 per arena)
        if total_trades < (valid_arenas * 0.5): 
            survival_score *= 0.5 # Stricter again (was 0.8) - Dead bots must die.
            
        # --- NEW: EQUITY RELIABILITY (User Request) ---
        # Penalize Negative AVG ROI heavily.
        if avg_roi < 0:
            survival_score *= 0.1 # 90% Penalty for losing money.
        # ----------------------------------------------
            
        # 3. Core Score: Sharpe + Sortino + ROI
        # Weighting: Reliability (60%), Growth (40%)
        # CLAMPING: Prevent Sharpe/Sortino inflation
        c_sharpe = min(avg_sharpe, 3.0) 
        c_sortino = min(avg_sortino, 5.0)
        
        # AGGRESSIVE BOOST: Scale ROI higher to reward ANY profit in bear markets
        # Add constant 1.0 base to raw_score to prevent multipication to zero if metrics are slightly negative
        # REPAIR: Base 10.0 to allow negative ROI to have a gradient.
        # If ROI is -0.5 (-50%), score = 10 + (-10) = 0.
        # If ROI is -0.1 (-10%), score = 10 + (-2) = 8.
        # If ROI is 0.1 (10%), score = 10 + 2 = 12.
        raw_score = 10.0 + (c_sharpe * 2.0) + (c_sortino * 1.0) + (avg_roi * 20.0)
        
        # 4. Quadratic Drawdown Penalty (RECALIBRATION)
        # Relaxed Curve: 
        # 10% DD = 1 / (1 + (0.5)^2) = ~0.8x
        # 40% DD = 1 / (1 + (2.0)^2) = 0.2x
        dd_penalty = 1.0 / (1.0 + (avg_dd * 5.0)**2) # Was * 10.0 (too harsh)

        
        # 5. Overfit Penalty: If Train ROI is 2x Val ROI
        # overfit_penalty = max(0, (train_roi - val_roi) * 2.0)
        
        # 6. Diversity Bonus: Must trade >= 3 assets if portfolio testing
        diversity_bonus = 1.0
        if len(active_symbols) >= 3 and valid_arenas < 3:
            diversity_bonus = 0.5 # Concentration Risk
            
        fitness = max(0.0, raw_score * survival_score * dd_penalty * diversity_bonus)
        
        # Fake Equity for Log readability (Projected)
        projected_equity = self.initial_capital * (1 + avg_roi)

        # === SANITY CHECK: Physics Violation ===
        # If Fitness is super-human (> 1000), it's likely a data glitch (bad tick).
        # Normal fitness is 10-100.
        if fitness > 1000.0:
            # logger.warning(f"⚠️ PHYSICS VIOLATION: Discarding Glitch Genome (Fit {fitness:.2f}, ROI {avg_roi*100:.1f}%)")
            return {
                'genome': genome,
                'fitness': 0.001, # Nuke it
                'roi': 0.0,
                'final_equity': self.initial_capital,
                'trades': 0, 'win_rate': 0, 'max_dd': 1.0, 'sharpe': 0, 'sortino': 0,
                'test_score': 0.0,
                'violation': True
            }

        return {
            'genome': genome,
            'final_equity': projected_equity, # For log display only
            'roi': avg_roi,
            'win_rate': win_rate,
            'trades': total_trades,
            'max_dd': avg_dd,
            'sharpe': avg_sharpe,
            'sortino': avg_sortino,
            'fitness': fitness,
            'validation_roi': 0.0, # Removed WF for speed/simplicity in V2
            'validation_trades': 0,
            'test_score': 1.0,
            'violation': False
        }

    def evolve(self, generations: int = 5, mutation_rate: float = 0.1, population_size: int = None, basket: List[str] = None) -> Dict:
        """
        Run the evolution loop.
        :param basket: Optional subset of assets to train on (Portfolio Level Testing)
        """
        target_pop_size = population_size if population_size else self.population_size
        basket_tag = "Full Market" if not basket else f"Basket({len(basket)})"
        
        # --- SANITY CHECK: Purge Glitched Champions ---
        if self.best_overall:
            # Re-run the gauntlet on the current champion to confirm its score with NEW logic
            # This kills 'Ghost' champions from previous buggy versions (e.g., Million Dollar Glitch)
            confirmed_stats = self.run_gauntlet(self.best_overall['genome'], basket=basket)
            
            # If the confirmed score is significantly worse, replace it
            if confirmed_stats['fitness'] < self.best_overall['fitness'] * 0.5:
                # logger.warning(f"📉 Downgrading Glitched Champion: Fit {self.best_overall['fitness']:.2f} -> {confirmed_stats['fitness']:.2f}")
                self.best_overall = confirmed_stats
        # ----------------------------------------------
        
        logger.info(f"🏹 Starting Evolution: {generations} Gens, {target_pop_size} Pop, {basket_tag}")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        for gen in range(generations):
            results = []
            violation_count = 0
            
            # --- PARALLEL GAUNTLET ---
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                future_to_genome = {executor.submit(self.run_gauntlet, g, basket=basket): g for g in self.population}
                
                for future in as_completed(future_to_genome):
                    try:
                        res = future.result()
                        results.append(res)
                        if res.get('violation', False):
                            violation_count += 1
                    except Exception as e:
                        logger.error(f"Genome evaluation failed: {e}")
            # ------------------------
            
            # --- EARLY STOPPING: Physics Violation Check ---
            violation_rate = violation_count / len(self.population)
            if violation_rate > 0.05:
                logger.warning(f"☢️ CRITICAL FAILURE: {violation_rate*100:.1f}% of population violated physics constraints. Triggering EMERGENCY RESET.")
                # Nuke population and restart with valid seeds
                self.seed_population()
                return self.evolve(generations=generations, mutation_rate=mutation_rate, population_size=population_size, basket=basket)
            # -----------------------------------------------

            results.sort(key=lambda x: x['fitness'], reverse=True)
            best_gen = results[0]
            
            # --- MEMORY: Update Hall of Fame ---
            # Keep top 10 unique genomes (Deduplication Logic)
            
            # 1. Add potential candidate
            self.hall_of_fame.append(best_gen)
            
            # 2. Sort by Fitness
            self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
            
            # 3. Deduplicate (JSON string based uniqueness)
            unique_hof = []
            seen_genomes = set()
            
            for entry in self.hall_of_fame:
                # Create hashable signature (sorted keys)
                g_sig = json.dumps(entry['genome'], sort_keys=True)
                if g_sig not in seen_genomes:
                    unique_hof.append(entry)
                    seen_genomes.add(g_sig)
            
            # 4. Truncate to Top 10
            self.hall_of_fame = unique_hof[:10]
            # -----------------------------------
            
            # === DIVERSITY CHECK ===
            # Calculate variance of a key gene (e.g., rsi_buy) to detect homogeneity
            rsi_values = [g['genome']['rsi_buy'] for g in results]
            pop_variance = np.var(rsi_values) if rsi_values else 0
            
            logger.info(f"Gen {gen+1} Leader: Fit {best_gen['fitness']:.2f} | Eq ${best_gen['final_equity']:.2f} (DD {best_gen['max_dd']*100:.1f}%) | Sharpe {best_gen.get('sharpe', 0):.2f} | Div {pop_variance:.1f}")
            
            if self.best_overall is None or best_gen['fitness'] > self.best_overall['fitness']:
                self.best_overall = best_gen
                
            # === PARETO RELOADED ===
            # Detect Stagnation
            is_stagnant = best_gen['fitness'] < 0.0001
            
            if is_stagnant:
                logger.warning(f"⚠️ EVOLUTION STALLED (Fit=0). Triggering MASS EXTINCTION event.")
                # Keep top 2 just in case, but flood the rest with Aliens
                elite_count = 2
                alien_count = target_pop_size - elite_count
            else:
                elite_count = max(2, int(target_pop_size * 0.20)) # Top 20%
                alien_count = int(target_pop_size * 0.20) # 20% Aliens

            survivors = results[:elite_count] 
            new_pop = [s['genome'] for s in survivors]
            
            # 1. 🛸 ALIENS - Escape Local Optima
            for _ in range(alien_count):
                new_pop.append(self.generate_random_genome())
                
            # 2. 🧬 CROSSOVER & MUTATION of the ELITES
            while len(new_pop) < target_pop_size:
                # Strictly breed from the Top 20%
                parent_a = random.choice(survivors)['genome']
                parent_b = random.choice(survivors)['genome']
                
                # Crossover (50/50 mix)
                child = {}
                for k in parent_a.keys():
                    child[k] = parent_a[k] if random.random() > 0.5 else parent_b[k]
                
                # Hyper-Mutation (Adapted to Mutation Rate)
                # mutation_rate ~ 0.15 means ~15% of genes mutate? 
                # Or just use it as a scaler for "Intensity". 
                # Let's say Intensity = 1 + (Rate * 10) -> 0.15 * 10 = 1.5 -> 1-2 genes.
                # If Rate is 0.35 -> 3.5 -> 3-4 genes.
                intensity_base = int(mutation_rate * 10)
                mutation_intensity = max(1, random.randint(1, intensity_base + 1))
                
                for _ in range(mutation_intensity):
                    key = random.choice(list(child.keys()))
                    if not isinstance(child[key], bool):
                        current_val = child[key]
                        
                        # Mutation Algo: Tanh Drift vs Proportional
                        # 50% chance of 'Nudge' (0.8-1.2x), 50% chance of 'Drift' (+/- small amount)
                        if random.random() > 0.5:
                            # Proportional Nudge
                            child[key] *= random.uniform(0.8, 1.2)
                        else:
                            # Drift (good for zero-bound values)
                            drift = random.uniform(-0.1, 0.1) * current_val
                            child[key] += drift
                            
                        # CRITICAL: Normalize
                        child[key] = self.normalize_gene(key, child[key])

                new_pop.append(child)
                
            self.population = new_pop
            
        logger.info(f"👑 Apex Predator Found: ${self.best_overall['final_equity']:.2f}")
        return self.best_overall

if __name__ == "__main__":
    # Test Run
    lab = EvolutionLab(leverage=5.0)
    lab.load_data()
    lab.seed_population()
    winner = lab.evolve(generations=3)
    print(winner)
