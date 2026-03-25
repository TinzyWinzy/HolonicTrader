"""
AEGIS QUANTSEC - Quick Integration Patch for main_live_phase4.py

This module provides a single function to integrate all AEGIS security components
into the existing HolonicTrader system.

Usage in main_live_phase4.py:
    from HolonicTrader.aegis_integration import initialize_aegis_security
    
    # After creating executor, governor, kraken_intel, ws clients, and RL agents:
    aegis = initialize_aegis_security(
        executor=executor,
        governor=governor,
        kraken_holon=kraken_intel,
        trader=trader,
        telegram_bot=overwatch.telegram if overwatch else None,
        chat_id=config.TELEGRAM_CHAT_ID,
        enable_all=True
    )
"""

import logging
from typing import Any, Optional, Dict

logger = logging.getLogger("AEGIS.Integration")


def initialize_aegis_security(
    executor: Any,
    governor: Any,
    kraken_holon: Any,
    trader: Optional[Any] = None,
    telegram_bot: Optional[Any] = None,
    chat_id: Optional[str] = None,
    enable_all: bool = True
) -> Dict[str, Any]:
    """
    Initialize all AEGIS QUANTSEC security components.
    
    Args:
        executor: ExecutorHolon instance
        governor: GovernorHolon instance
        kraken_holon: KrakenHolon instance
        trader: TraderHolon instance (optional, for RL agent wrapping)
        telegram_bot: Telegram bot instance (optional, for alerts)
        chat_id: Telegram chat ID (optional, for alerts)
        enable_all: Enable all security features (default True)
    
    Returns:
        Dictionary with references to all AEGIS components
    """
    from HolonicTrader.log_integrity import LogIntegrityManager, IntegrityAlertHandler
    from HolonicTrader.position_reconciliation import integrate_reconciliation_engine
    from HolonicTrader.timestamp_oracle import integrate_websocket_monitor, integrate_timestamp_oracle
    from HolonicTrader.rl_agent_security import wrap_dqn_agent, wrap_ppo_agent
    
    aegis_components = {
        'enabled': False,
        'log_manager': None,
        'reconciliation_engine': None,
        'websocket_monitor': None,
        'timestamp_oracle': None,
        'rl_security': {}
    }
    
    try:
        print(">> ==========================================")
        print(">>    AEGIS QUANTSEC SECURITY FRAMEWORK      ")
        print(">> ==========================================")
        
        # ======================================================================
        # PHASE 1: Log Integrity Engine
        # ======================================================================
        print(">> [AEGIS Phase 1] Initializing Log Integrity Engine...")
        log_manager = LogIntegrityManager(
            storage_path="logs/execution_integrity.json",
            auto_anchor_interval=100,
            enable_tamper_detection=True
        )
        aegis_components['log_manager'] = log_manager
        
        # Add Telegram alerts for log integrity
        if telegram_bot and chat_id:
            log_alerts = IntegrityAlertHandler(
                integrity_manager=log_manager,
                telegram_bot=telegram_bot,
                chat_id=chat_id
            )
            print(">> [AEGIS Phase 1] Telegram alerts enabled")
        
        # Log system start
        log_manager.log_event(
            event_type="SYSTEM_START",
            symbol="SYSTEM",
            data={
                'components': ['log_integrity', 'reconciliation', 'timestamp_oracle', 'rl_security'],
                'version': '1.0'
            }
        )
        
        # ======================================================================
        # PHASE 2: Position Reconciliation Engine
        # ======================================================================
        print(">> [AEGIS Phase 2] Initializing Position Reconciliation Engine...")
        recon_engine = integrate_reconciliation_engine(
            executor_holon=executor,
            kraken_holon=kraken_holon,
            websocket_monitor=None,  # Will be set below
            enable_telegram=bool(telegram_bot and chat_id),
            telegram_bot=telegram_bot,
            chat_id=chat_id,
            auto_resolve=False  # Set to True for auto-purge of leak positions
        )
        recon_engine.start()
        aegis_components['reconciliation_engine'] = recon_engine
        
        # ======================================================================
        # PHASE 3: Timestamp Oracle & Websocket Integrity
        # ======================================================================
        print(">> [AEGIS Phase 3] Initializing Timestamp Oracle...")
        timestamp_oracle = integrate_timestamp_oracle()
        aegis_components['timestamp_oracle'] = timestamp_oracle
        
        # Get websocket clients from observer holons if available
        ws_client = None
        if hasattr(trader, 'sub_holons') and 'observer' in trader.sub_holons:
            observer = trader.sub_holons['observer']
            if hasattr(observer, 'ws_client'):
                ws_client = observer.ws_client
        
        print(">> [AEGIS Phase 3] Initializing Websocket Integrity Monitor...")
        ws_monitor = integrate_websocket_monitor(
            ws_client=ws_client,
            kraken_holon=kraken_holon,
            enable_alerts=bool(telegram_bot and chat_id),
            telegram_bot=telegram_bot,
            chat_id=chat_id
        )
        aegis_components['websocket_monitor'] = ws_monitor
        
        # Link websocket monitor to reconciliation engine
        if recon_engine:
            recon_engine.ws_monitor = ws_monitor
            print(">> [AEGIS Phase 3] Websocket monitor linked to reconciliation engine")
        
        # ======================================================================
        # PHASE 4: RL Agent Security Wrapper
        # ======================================================================
        print(">> [AEGIS Phase 4] Initializing RL Agent Security...")
        
        if trader and hasattr(trader, 'sub_holons'):
            # Wrap DQN agent if present
            if 'dqn' in trader.sub_holons:
                print(">> [AEGIS Phase 4] Wrapping DQN agent with security layer...")
                secured_dqn = wrap_dqn_agent(
                    trader.sub_holons['dqn'],
                    enable_all_features=enable_all
                )
                trader.sub_holons['dqn'] = secured_dqn
                aegis_components['rl_security']['dqn'] = secured_dqn
            
            # Wrap PPO agent if present
            if 'ppo' in trader.sub_holons:
                print(">> [AEGIS Phase 4] Wrapping PPO agent with security layer...")
                secured_ppo = wrap_ppo_agent(
                    trader.sub_holons['ppo'],
                    enable_all_features=enable_all
                )
                trader.sub_holons['ppo'] = secured_ppo
                aegis_components['rl_security']['ppo'] = secured_ppo
        
        # ======================================================================
        # INJECT INTO EXECUTOR FOR EASY ACCESS
        # ======================================================================
        executor._aegis_components = aegis_components
        
        # Add helper method to executor
        def get_aegis_report():
            """Get comprehensive AEGIS security report."""
            report = {
                'log_integrity': log_manager.get_integrity_report() if log_manager else None,
                'position_reconciliation': recon_engine.get_latest_report().to_dict() if recon_engine and recon_engine.get_latest_report() else None,
                'websocket_integrity': ws_monitor.get_integrity_report().to_dict() if ws_monitor else None,
                'rl_security': {}
            }
            
            if 'dqn' in aegis_components['rl_security']:
                report['rl_security']['dqn'] = aegis_components['rl_security']['dqn'].get_security_report().to_dict()
            if 'ppo' in aegis_components['rl_security']:
                report['rl_security']['ppo'] = aegis_components['rl_security']['ppo'].get_security_report().to_dict()
            
            return report
        
        executor.get_aegis_report = get_aegis_report
        
        # ======================================================================
        # SUMMARY
        # ======================================================================
        aegis_components['enabled'] = True
        
        print(">> ==========================================")
        print(">>    AEGIS QUANTSEC ONLINE                  ")
        print(">> ==========================================")
        print(">> Components:")
        print(">>   - Phase 1: Log Integrity Engine ✓")
        print(">>   - Phase 2: Position Reconciliation ✓")
        print(">>   - Phase 3: Timestamp Oracle + WS Monitor ✓")
        print(">>   - Phase 4: RL Agent Security ✓")
        print(">>")
        print(">> Security Coverage:")
        print(">>   - Log tampering detection")
        print(">>   - Ghost/Leak position detection")
        print(">>   - Websocket sequence validation")
        print(">>   - Reward poisoning protection")
        print(">> ==========================================")
        
        # Log initialization
        log_manager.log_event(
            event_type="AEGIS_INIT",
            symbol="SYSTEM",
            data={
                'phase1': True,
                'phase2': True,
                'phase3': True,
                'phase4': True,
                'telegram_alerts': bool(telegram_bot and chat_id)
            }
        )
        
    except Exception as e:
        print(f">> [AEGIS] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        aegis_components['enabled'] = False
    
    return aegis_components


def get_aegis_status(executor: Any) -> Dict[str, Any]:
    """
    Get current AEGIS security status.
    
    Args:
        executor: ExecutorHolon instance
    
    Returns:
        Status dictionary
    """
    if not hasattr(executor, '_aegis_components') or not executor._aegis_components.get('enabled'):
        return {'status': 'NOT_INITIALIZED'}
    
    components = executor._aegis_components
    status = {
        'status': 'ACTIVE',
        'components': {}
    }
    
    # Log integrity
    if components.get('log_manager'):
        log_report = components['log_manager'].get_integrity_report()
        status['components']['log_integrity'] = {
            'status': log_report['status'],
            'entries': log_report['total_entries']
        }
    
    # Position reconciliation
    if components.get('reconciliation_engine'):
        recon = components['reconciliation_engine']
        report = recon.get_latest_report()
        if report:
            status['components']['position_reconciliation'] = {
                'status': report.summary['status'],
                'integrity_score': recon.get_integrity_score()
            }
    
    # Websocket integrity
    if components.get('websocket_monitor'):
        ws_report = components['websocket_monitor'].get_integrity_report()
        status['components']['websocket_integrity'] = {
            'health': ws_report.health_status
        }
    
    # RL security
    status['components']['rl_security'] = {}
    if 'dqn' in components['rl_security']:
        dqn_report = components['rl_security']['dqn'].get_security_report()
        status['components']['rl_security']['dqn'] = {
            'status': dqn_report.status,
            'security_score': dqn_report.security_score
        }
    if 'ppo' in components['rl_security']:
        ppo_report = components['rl_security']['ppo'].get_security_report()
        status['components']['rl_security']['ppo'] = {
            'status': ppo_report.status,
            'security_score': ppo_report.security_score
        }
    
    return status
