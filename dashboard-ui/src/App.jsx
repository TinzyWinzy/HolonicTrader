import React, { useState } from 'react';
import { SocketProvider, useSocket } from './context/SocketContext';
import DashboardLayout from './components/DashboardLayout';
import LiveLog from './components/LiveLog';
import StatCard from './components/StatCard';
import EquityChart from './components/EquityChart';
import RadarPanel from './components/RadarPanel';
import SignalsPanel from './components/SignalsPanel';
import EvolutionPanel from './components/EvolutionPanel';
import OrderFlowPanel from './components/OrderFlowPanel';
import PositionsPanel from './components/PositionsPanel';
import DefconIndicator from './components/DefconIndicator';
import ArbitragePanel from './components/ArbitragePanel';
import MonteCarloPanel from './components/MonteCarloPanel';
import ConfigPanel from './components/ConfigPanel';
import TradePanel from './components/TradePanel';
import PositionHealthPanel from './components/PositionHealthPanel';
import SitrepPanel from './components/SitrepPanel';
import { Activity, Shield, Wallet, Play, Square, TrendingUp } from 'lucide-react';
import clsx from 'clsx';
import PortfolioAnalyticsPanel from './components/PortfolioAnalyticsPanel';
import SystemAlerts from './components/SystemAlerts';

function Dashboard() {
  const { isConnected, systemState, sendCommand } = useSocket();
  const [loadingCmd, setLoadingCmd] = useState(null);
  const [activeTab, setActiveTab] = useState('market');

  // Flat State Mapping
  const state = systemState || {};
  const {
    equity = 0,
    pnl = 0,
    health_score: health = 0,
    system_status: status = 'DISCONNECTED',
    regime = '---',
    positions = [],
    portfolio_health: portfolioHealth = {},
    doomsday = { defcon_level: 5, crisis_active: false },
    arbitrage = [],
    evolution = {},
    order_flow: orderFlow = {},
    monte_carlo: monteCarlo = {},
    radar = [],
    logs = [],
    equity_history = [],
    last_scan: lastScan
  } = state;

  const managementSignals = radar.filter(s => s?.metadata?.strategy === 'POSITION_MANAGEMENT');
  const isRunning = ['SOLVENT', 'RUNNING', 'ACTIVE'].includes(status);

  const handleCmd = async (cmd) => {
    setLoadingCmd(cmd);
    if (cmd === 'stop') {
      try {
        await fetch('/api/emergency/panic', { method: 'POST' });
      } catch (e) {
        console.error("PANIC NETWORK ERROR:", e);
      }
    } else {
      await sendCommand(cmd, { symbol: 'BTC/USDT', leverage: 5.0, allocation: 0.1 });
    }
    setTimeout(() => setLoadingCmd(null), 1000);
  };

  return (
    <DashboardLayout>
      <SystemAlerts />

      {/* --- COLUMN 1: CONTROLS & METRICS --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-3 flex flex-col gap-4 h-full min-h-0">
        {/* Status Group */}
        <StatusBadge status={status} health={health} />

        <div className="grid grid-cols-2 gap-3">
          <StatCard
            label="TOTAL EQUITY"
            value={`$${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
            icon={Wallet}
            color="text-emerald-400"
          />
          <StatCard
            label="SESSION PNL"
            value={`${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`}
            icon={TrendingUp}
            color={pnl >= 0 ? "text-emerald-400" : "text-red-400"}
          />
        </div>

        <DefconIndicator doomsday={doomsday} />
        <ArbitragePanel opportunities={arbitrage} />

        {/* Fill remaining height with log terminal */}
        <div className="flex-1 flex flex-col min-h-[300px]">
          <LiveLog logs={logs} />
        </div>

        {/* Action Controls */}
        <div className="bg-holon-card p-3 rounded-xl border border-slate-700/50 flex flex-col gap-2 mt-auto">
          <ActionButton
            label={isRunning ? "SYSTEM ACTIVE" : "INITIATE"}
            icon={Play}
            onClick={() => handleCmd('start')}
            active={isRunning}
            loading={loadingCmd === 'start'}
            disabled={isRunning}
            color="emerald"
            fullWidth
          />
          <ActionButton
            label="HALT OPERATIONS"
            icon={Square}
            onClick={() => handleCmd('stop')}
            loading={loadingCmd === 'stop'}
            disabled={status === 'STOPPED'}
            color="red"
            fullWidth
          />
        </div>
      </div>

      {/* --- COLUMN 2: HUB INTERFACE (6 Cols) --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-6 flex flex-col gap-4 h-full min-h-0">
        {/* Tab Navigation */}
        <div className="flex gap-1 p-1 bg-slate-900/40 rounded-lg border border-slate-800/50 shrink-0">
          {['market', 'sitrep', 'analytics', 'signals', 'evolution', 'order-flow'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                "flex-1 py-2 text-[10px] font-orbitron tracking-widest rounded transition-all transform active:scale-95",
                activeTab === tab
                  ? "bg-slate-700 text-holon-accent shadow-lg border border-slate-600/50"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
              )}
            >
              {tab.replace('-', ' ').toUpperCase()}
            </button>
          ))}
        </div>

        {/* Tab Content Area */}
        <div className="flex-1 flex flex-col gap-4 min-h-0 overflow-y-auto pr-1">
          {activeTab === 'market' && (
            <>
              <div className="h-64 shrink-0 bg-holon-card rounded-xl border border-slate-700/50 p-4 relative flex flex-col shadow-lg">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-bold text-white font-orbitron tracking-wider">EQUITY CURVE</h3>
                  <p className="text-[10px] text-emerald-400 font-mono animate-pulse">● LIVE FEED</p>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <EquityChart dataPoints={equity_history} />
                </div>
              </div>

              <div className="flex-1 min-h-[350px]">
                <RadarPanel items={radar} />
              </div>
            </>
          )}

          {activeTab === 'analytics' && <PortfolioAnalyticsPanel />}
          {activeTab === 'signals' && <SignalsPanel signals={radar} lastScan={lastScan} />}
          {activeTab === 'evolution' && <EvolutionPanel data={evolution} />}
          {activeTab === 'sitrep' && <SitrepPanel />}
          {activeTab === 'order-flow' && <OrderFlowPanel data={orderFlow} />}
        </div>
      </div>

      {/* --- COLUMN 3: POSITIONS & INTELLIGENCE (3 Cols) --- */}
      <div className="col-span-1 md:col-span-12 lg:col-span-3 flex flex-col gap-4 h-full min-h-0">
        {/* Regime Badge */}
        <div className="bg-gradient-to-r from-blue-900/40 to-slate-900/40 p-3 rounded-xl border border-blue-800/30 flex items-center justify-between shrink-0 shadow-lg">
          <span className="text-[10px] font-mono text-blue-300 uppercase tracking-wider">Market Regime</span>
          <span className="font-bold font-orbitron text-sm text-blue-100 drop-shadow-md">{regime}</span>
        </div>

        {/* Intelligence Widgets */}
        <div className="grid grid-cols-1 gap-4 shrink-0">
          <PositionHealthPanel managementSignals={managementSignals} />
          <MonteCarloPanel data={monteCarlo} />
        </div>

        {/* Positions Table (Takes remaining height) */}
        <div className="flex-1 min-h-[300px]">
          <PositionsPanel positions={positions} portfolioHealth={portfolioHealth} />
        </div>

        {/* Config / Trade Panel Accordions */}
        <div className="shrink-0 flex flex-col gap-2">
          <TradePanel />
          <ConfigPanel />
        </div>
      </div>
    </DashboardLayout>
  );
}

// --- Helper Components ---

const StatusBadge = ({ status, health }) => {
  const isRunning = ['SOLVENT', 'RUNNING', 'ACTIVE'].includes(status);
  return (
    <div className="bg-holon-card p-4 rounded-xl border border-slate-700/50 flex items-center gap-4 relative overflow-hidden shadow-lg group">
      <div className={clsx("h-2 w-2 rounded-full absolute top-3 right-3 animate-pulse shadow-[0_0_10px_currentColor]", isRunning ? "bg-emerald-500 text-emerald-500" : "bg-red-500 text-red-500")} />
      <div className={clsx("p-3 rounded-lg transition-colors", isRunning ? "bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20" : "bg-red-500/10 text-red-500")}>
        <Activity size={24} />
      </div>
      <div>
        <p className="text-[10px] font-mono text-holon-dim uppercase tracking-wider">SYSTEM STATUS</p>
        <p className={clsx("text-lg font-bold font-orbitron tracking-wide", isRunning ? "text-emerald-400 text-shadow-glow" : "text-red-400")}>{status}</p>
        <div className="flex items-center gap-2 mt-1">
          <div className="h-1 w-16 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-purple-500 transition-all duration-500 shadow-[0_0_8px_#a855f7]" style={{ width: `${health}%` }} />
          </div>
          <span className="text-[9px] text-purple-400 font-mono">{health.toFixed(0)}% HP</span>
        </div>
      </div>
    </div>
  );
};

const ActionButton = ({ label, icon: Icon, onClick, loading, disabled, color, active, fullWidth }) => {
  const baseClass = "flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-bold font-orbitron tracking-wider text-[10px] transition-all shadow-lg active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed border h-10";
  const colors = {
    emerald: "bg-emerald-600/20 text-emerald-400 border-emerald-500/50 hover:bg-emerald-500/30 hover:shadow-emerald-500/20",
    red: "bg-red-600/20 text-red-400 border-red-500/50 hover:bg-red-500/30 hover:shadow-red-500/20"
  };

  return (
    <button onClick={onClick} disabled={disabled || loading} className={clsx(baseClass, colors[color], fullWidth && "w-full")}>
      {loading ? <span className="animate-spin">⏳</span> : <Icon size={14} fill={active ? "currentColor" : "none"} />}
      {label}
    </button>
  );
};

export default function App() {
  return (
    <SocketProvider>
      <Dashboard />
    </SocketProvider>
  );
}
