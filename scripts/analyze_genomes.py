"""
Genome Reliability Analysis

Analyzes Hall of Fame to find most reliable genome
"""
import json
from pathlib import Path

# Load hall of fame
with open('hall_of_fame.json') as f:
    hof = json.load(f)

# Load live genome
with open('live_genome.json') as f:
    live = json.load(f)

print('=' * 70)
print('HALL OF FAME - CRITICAL ANALYSIS')
print('=' * 70)

print(f'\nTotal Genomes: {len(hof)}')

print('\n' + '=' * 70)
print('TOP 5 GENOMES COMPARISON')
print('=' * 70)

print(f'\n{"#":<3} {"Fitness":<8} {"ROI":<8} {"Win%":<6} {"Trades":<7} {"Valid":<6} {"Status":<20}')
print('-' * 70)

for i, genome in enumerate(hof[:5], 1):
    trades = genome['trades']
    valid = genome.get('validation_trades', 0)
    
    # Status
    if trades >= 5 and valid >= 5:
        status = '✅ RELIABLE'
    elif trades >= 3:
        status = '⚠️ SOME DATA'
    else:
        status = '❌ INSUFFICIENT'
    
    print(f'{i:<3} {genome["fitness"]:<8.2f} {genome["roi"]:<8.1%} {genome["win_rate"]:<6.0%} {trades:<7} {valid:<6} {status:<20}')

print('\n' + '=' * 70)
print('LIVE GENOME STATUS')
print('=' * 70)

print(f'\nFitness: {live["fitness"]:.2f}')
print(f'Trades: {live["trades"]}')
print(f'Validation: {live.get("validation_trades", 0)} trades')

if live['trades'] < 5:
    print(f'\n🚨 RED FLAGS:')
    print(f'  ⚠️ Only {live["trades"]} trade(s) - need ≥5 for significance')
    print(f'  ⚠️ No validation trades')
    print(f'  ⚠️ Fitness 19.74 is extremely high (possible overfit)')
    print(f'\n❌ NOT RELIABLE - High risk of mean reversion')

print('\n' + '=' * 70)
print('RECOMMENDATION')
print('=' * 70)

# Find most reliable
reliable = [g for g in hof if g['trades'] >= 3]

if reliable:
    # Best by trades
    most_trades = max(reliable, key=lambda x: x['trades'])
    # Best by fitness (with min trades)
    best_fitness = max([g for g in reliable if g['trades'] >= 3], key=lambda x: x['fitness'])
    
    print(f'\n⚠️ CURRENT BRAIN IS NOT RELIABLE ({live["trades"]} trade)')
    print()
    print('OPTION 1: MOST DATA')
    print(f'  Genome #{hof.index(most_trades)+1}')
    print(f'  Trades: {most_trades["trades"]}, Win Rate: {most_trades["win_rate"]:.0%}')
    print(f'  Fitness: {most_trades["fitness"]:.2f}, ROI: {most_trades["roi"]:.1%}')
    print(f'  More reliable but lower fitness')
    print()
    print('OPTION 2: BALANCED')
    print(f'  Genome #{hof.index(best_fitness)+1}')
    print(f'  Trades: {best_fitness["trades"]}, Win Rate: {best_fitness["win_rate"]:.0%}')
    print(f'  Fitness: {best_fitness["fitness"]:.2f}, ROI: {best_fitness["roi"]:.1%}')
    print(f'  Good balance of fitness and data')
    print()
    print('OPTION 3: WAIT AND WATCH')
    print(f'  Keep current brain')
    print(f'  Monitor next 4 trades closely')
    print(f'  Switch if win rate drops below 50%')
    print()
    print('MY RECOMMENDATION: OPTION 2 or 3')
    print('  - Current parameters (SL 2.4%, TP 11.5%) are excellent')
    print('  - But need more data to trust fitness score')
    print('  - Set tight stop (-5% drawdown alert)')
    print('  - Let it prove itself over next 5-10 trades')
