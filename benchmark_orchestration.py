"""
HolonicTrader Model Benchmark Orchestration Script

Automates benchmarking, drift detection, and retraining recommendation for all models.

Steps:
1. Inventory models and last training date
2. Run backtests on recent data
3. Compare live vs. simulated performance
4. Analyze feature drift/regime change
5. Evaluate prediction error
6. Output benchmark report and retrain recommendations

Run from HolonicTrader root directory.
"""
import os
import json
import glob
import subprocess
from datetime import datetime

# --- Config ---
MODEL_FILES = {
    'DQN': 'dqn_model.keras',
    'XGBoost': 'xgboost_model.json',
}
BACKTEST_SCRIPT = 'scripts/ab_backtest.py'
ANALYZE_WINNERS_SCRIPT = 'analyze_winners.py'
TRAINING_SCRIPTS = {
    'DQN': 'scripts/train_quick_model.py',
    'XGBoost': 'scripts/train_xgboost.py',
}
DATA_DIR = 'datasets/'
BACKTEST_RESULTS = 'backtests/ab_backtest.json'

# --- Helpers ---
def get_model_info():
    info = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            ts = os.path.getmtime(path)
            info[name] = {
                'file': path,
                'last_trained': datetime.fromtimestamp(ts).isoformat(),
            }
        else:
            info[name] = {'file': path, 'last_trained': None}
    return info

def run_backtest():
    print('Running backtest...')
    result = subprocess.run(['python', BACKTEST_SCRIPT], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print('Backtest failed:', result.stderr)
    return os.path.exists(BACKTEST_RESULTS)

def analyze_winners():
    print('Analyzing winners...')
    result = subprocess.run(['python', ANALYZE_WINNERS_SCRIPT, '--json'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print('Analysis failed:', result.stderr)
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}

def check_feature_drift():
    # Placeholder: implement regime/entropy drift check if available
    print('Checking regime/feature drift...')
    # Could call regime engine or analyze entropy logs
    return {'drift_detected': False, 'details': 'Not implemented'}

def evaluate_prediction_error():
    print('Evaluating prediction error...')
    errors = {}
    try:
        with open('logs/train_rich.json', 'r') as f:
            rich = json.load(f)
            errors['train_rich_mse'] = rich.get('mse')
            errors['train_rich_n_train'] = rich.get('n_train')
            errors['train_rich_n_test'] = rich.get('n_test')
    except Exception as e:
        errors['train_rich_error'] = str(e)
    try:
        with open('logs/train_quick.json', 'r') as f:
            quick = json.load(f)
            errors['train_quick_mse'] = quick.get('mse')
            errors['train_quick_n_train'] = quick.get('n_train')
            errors['train_quick_n_test'] = quick.get('n_test')
    except Exception as e:
        errors['train_quick_error'] = str(e)
    return errors

def main():
    report = {}
    report['model_info'] = get_model_info()
    if run_backtest():
        with open(BACKTEST_RESULTS, 'r') as f:
            report['backtest'] = json.load(f)
    else:
        report['backtest'] = None
    report['live_vs_sim'] = analyze_winners()
    report['feature_drift'] = check_feature_drift()
    report['prediction_error'] = evaluate_prediction_error()
    # Decision logic
    retrain = False
    reasons = []
    if report['backtest'] and report['backtest'].get('expectancy', 0) < 0.1:
        retrain = True
        reasons.append('Low expectancy in backtest')
    if report['feature_drift'].get('drift_detected'):
        retrain = True
        reasons.append('Feature/regime drift detected')
    if report['prediction_error'].get('error') and report['prediction_error']['error'] > 0.2:
        retrain = True
        reasons.append('High prediction error')
    report['retrain_recommended'] = retrain
    report['retrain_reasons'] = reasons
    with open('benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('\n=== BENCHMARK REPORT ===')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
