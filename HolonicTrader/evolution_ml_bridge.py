"""
Evolution-ML Bridge

Integrates Evolutionary Brain Transplants with ML Quality Control

Purpose:
- Validate new genomes with ML before deployment
- Ensure evolved parameters align with ML predictions
- Prevent overfit genomes from going live
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class EvolutionMLBridge:
    """
    Bridge between Evolution Engine and ML System
    
    Validates brain transplants before deployment
    """
    
    def __init__(self):
        self.hall_of_fame_path = Path('hall_of_fame.json')
        self.live_genome_path = Path('live_genome.json')
        self.ml_advisor = None
        
        # Validation thresholds
        self.min_trades_for_validation = 5
        self.min_validation_roi = 0.0  # Must be positive
        self.max_fitness_inflation = 100.0  # Sanity check
        
        print("🧬 Evolution-ML Bridge initialized")
    
    def validate_brain_transplant(self, new_genome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate new evolved genome before deployment
        
        Returns:
        {
            'approved': bool,
            'reason': str,
            'risk_level': 'LOW' | 'MEDIUM' | 'HIGH',
            'ml_alignment': float
        }
        """
        result = {
            'approved': False,
            'reason': '',
            'risk_level': 'HIGH',
            'ml_alignment': 0.0
        }
        
        # Check 1: Statistical Significance
        trades = new_genome.get('trades', 0)
        if trades < self.min_trades_for_validation:
            result['reason'] = f'Insufficient trades ({trades} < {self.min_trades_for_validation})'
            result['risk_level'] = 'HIGH'
            return result
        
        # Check 2: Validation Performance
        validation_roi = new_genome.get('validation_roi', 0.0)
        validation_trades = new_genome.get('validation_trades', 0)
        
        if validation_trades > 0 and validation_roi < self.min_validation_roi:
            result['reason'] = f'Negative validation ROI ({validation_roi:.1%})'
            result['risk_level'] = 'HIGH'
            return result
        
        # Check 3: Fitness Sanity Check
        fitness = new_genome.get('fitness', 0.0)
        if fitness > self.max_fitness_inflation:
            result['reason'] = f'Fitness inflation detected ({fitness:.1f} > {self.max_fitness_inflation})'
            result['risk_level'] = 'MEDIUM'
            return result
        
        # Check 4: Parameter Sanity
        genome = new_genome.get('genome', {})
        
        # RSI thresholds must be reasonable
        rsi_buy = genome.get('rsi_buy', 50)
        rsi_sell = genome.get('rsi_sell', 50)
        
        if rsi_buy >= rsi_sell:
            result['reason'] = f'Invalid RSI (buy {rsi_buy:.1f} >= sell {rsi_sell:.1f})'
            result['risk_level'] = 'HIGH'
            return result
        
        if rsi_buy < 10 or rsi_buy > 50:
            result['reason'] = f'RSI buy out of range ({rsi_buy:.1f})'
            result['risk_level'] = 'MEDIUM'
            return result
        
        if rsi_sell < 50 or rsi_sell > 90:
            result['reason'] = f'RSI sell out of range ({rsi_sell:.1f})'
            result['risk_level'] = 'MEDIUM'
            return result
        
        # Stop loss and take profit sanity
        stop_loss = genome.get('stop_loss', 0.0)
        take_profit = genome.get('take_profit', 0.0)
        
        if stop_loss <= 0 or stop_loss > 0.2:  # 0-20%
            result['reason'] = f'Invalid stop loss ({stop_loss:.1%})'
            result['risk_level'] = 'HIGH'
            return result
        
        if take_profit <= 0 or take_profit > 0.5:  # 0-50%
            result['reason'] = f'Invalid take profit ({take_profit:.1%})'
            result['risk_level'] = 'MEDIUM'
            return result
        
        # RR ratio check
        rr_ratio = take_profit / stop_loss if stop_loss > 0 else 0
        if rr_ratio < 1.0:  # Must have at least 1:1 RR
            result['reason'] = f'Poor RR ratio ({rr_ratio:.2f}:1)'
            result['risk_level'] = 'MEDIUM'
            return result
        
        # All checks passed
        result['approved'] = True
        result['reason'] = 'All validation checks passed'
        result['risk_level'] = 'LOW' if trades >= 10 else 'MEDIUM'
        result['ml_alignment'] = self._calculate_ml_alignment(genome)
        
        return result
    
    def _calculate_ml_alignment(self, genome: Dict[str, Any]) -> float:
        """
        Calculate how well evolved parameters align with ML predictions
        
        Returns: 0.0-1.0 (1.0 = perfect alignment)
        """
        # TODO: Integrate with ML Advisor
        # For now, return based on parameter quality
        
        score = 1.0
        
        # Penalize extreme parameters
        rsi_buy = genome.get('rsi_buy', 30)
        rsi_sell = genome.get('rsi_sell', 70)
        
        # Ideal RSI range: 25-35 buy, 65-75 sell
        if not (25 <= rsi_buy <= 35):
            score -= 0.1
        if not (65 <= rsi_sell <= 75):
            score -= 0.1
        
        # Ideal RR ratio: 2:1 or better
        stop_loss = genome.get('stop_loss', 0.02)
        take_profit = genome.get('take_profit', 0.04)
        rr_ratio = take_profit / stop_loss if stop_loss > 0 else 0
        
        if rr_ratio >= 2.0:
            score += 0.1  # Bonus for good RR
        elif rr_ratio < 1.5:
            score -= 0.2  # Penalty for poor RR
        
        return max(0.0, min(1.0, score))
    
    def get_hall_of_fame_stats(self) -> Dict[str, Any]:
        """Get statistics about Hall of Fame genomes"""
        
        if not self.hall_of_fame_path.exists():
            return {'error': 'Hall of Fame not found'}
        
        with open(self.hall_of_fame_path) as f:
            hof = json.load(f)
        
        if not isinstance(hof, list) or len(hof) == 0:
            return {'error': 'Hall of Fame is empty'}
        
        # Calculate stats
        fitnesses = [g.get('fitness', 0) for g in hof]
        rois = [g.get('roi', 0) for g in hof]
        trades = [g.get('trades', 0) for g in hof]
        win_rates = [g.get('win_rate', 0) for g in hof]
        
        return {
            'total_genomes': len(hof),
            'fitness': {
                'avg': sum(fitnesses) / len(fitnesses),
                'max': max(fitnesses),
                'min': min(fitnesses)
            },
            'roi': {
                'avg': sum(rois) / len(rois),
                'max': max(rois),
                'min': min(rois)
            },
            'trades': {
                'avg': sum(trades) / len(trades),
                'max': max(trades),
                'min': min(trades)
            },
            'win_rate': {
                'avg': sum(win_rates) / len(win_rates),
                'best': max(win_rates)
            },
            'validation_concerns': sum(1 for g in hof if g.get('validation_trades', 0) == 0)
        }
    
    def recommend_ensemble_size(self) -> int:
        """
        Recommend how many top genomes to use in ensemble
        
        Returns: 3-10 based on Hall of Fame quality
        """
        stats = self.get_hall_of_fame_stats()
        
        if 'error' in stats:
            return 3  # Default
        
        # More genomes if we have quality diversity
        if stats['total_genomes'] >= 10 and stats['validation_concerns'] < 3:
            return 5  # Use top 5
        
        if stats['total_genomes'] >= 5:
            return 3  # Use top 3
        
        return min(stats['total_genomes'], 3)  # Use what we have


# Singleton
_bridge_instance = None

def get_evolution_ml_bridge() -> EvolutionMLBridge:
    """Get bridge singleton"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = EvolutionMLBridge()
    return _bridge_instance


# Convenience function
def validate_genome(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Validate evolved genome"""
    bridge = get_evolution_ml_bridge()
    return bridge.validate_brain_transplant(genome)


print("🧬 Evolution-ML Bridge loaded")
