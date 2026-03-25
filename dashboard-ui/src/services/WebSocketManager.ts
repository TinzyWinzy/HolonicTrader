import { io, Socket } from 'socket.io-client';
import { useMarketStore } from '../store/useMarketStore';
import type { HubState } from '../types/market';

// ─── WebSocket Manager ─────────────────────────────────────────────────────────
// Singleton that connects to signal_server.py and pipes data into Zustand.
// In dev mode, uses Vite proxy (same origin). Set VITE_API_URL for production.

class WebSocketManager {
    private socket: Socket | null = null;
    private static instance: WebSocketManager | null = null;

    static getInstance(): WebSocketManager {
        if (!WebSocketManager.instance) {
            WebSocketManager.instance = new WebSocketManager();
        }
        return WebSocketManager.instance;
    }

    private get baseUrl(): string {
        // @ts-ignore
        return import.meta.env.VITE_API_URL || 'http://localhost:5000';
    }

    connect(): void {
        if (this.socket?.connected) return;

        const url = this.baseUrl;
        console.log(`[WS] Connecting to ${url}...`);

        this.socket = io(url, {
            transports: ['websocket', 'polling'], // Fallback to polling if websocket fails
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 20000,
        });

        this.socket.on('connect', () => {
            console.log('[WS] Connected to signal_server');
            useMarketStore.setState({ isConnected: true });
            this.fetchInitialState();
        });

        this.socket.on('disconnect', () => {
            console.log('[WS] Disconnected');
            useMarketStore.setState({ isConnected: false });
        });

        // Primary data channel — flat hub_state payload every ~1s
        this.socket.on('hub_state', (data: Partial<HubState>) => {
            useMarketStore.getState().updateFromHubState(data);
        });

        // Signal updates (live report from scanner)
        this.socket.on('signals_update', (data: { signals: any[]; time: string }) => {
            if (data.signals?.length) {
                useMarketStore.setState({ radar: data.signals });
            }
        });

        // State update (legacy)
        this.socket.on('state_update', (data: Partial<HubState>) => {
            useMarketStore.getState().updateFromHubState(data);
        });

        // System alerts
        this.socket.on('system_alert', (alert: { level: string; message: string; timestamp: number }) => {
            console.log(`[ALERT:${alert.level}] ${alert.message}`);
        });
    }

    private async fetchInitialState(): Promise<void> {
        try {
            const base = this.baseUrl;
            const [hubRes, signalsRes] = await Promise.all([
                fetch(`${base}/api/hub/state`).then(r => r.json()),
                fetch(`${base}/api/signals`).then(r => r.json()),
            ]);

            const dashboard = hubRes.status_files?.dashboard || {};
            const signals = signalsRes.signals || [];

            useMarketStore.getState().updateFromHubState({
                radar: signals,
                equity: dashboard.equity || dashboard.balance || 0,
                regime: dashboard.regime,
                system_status: dashboard.solvency_status,
            } as Partial<HubState>);
        } catch (err) {
            console.error('[WS] Initial fetch failed:', err);
        }
    }

    async sendCommand(cmd: string, payload: Record<string, unknown> = {}): Promise<any> {
        const base = this.baseUrl;

        if (cmd === 'panic') {
            return fetch(`${base}/api/emergency/panic`, { method: 'POST' }).then(r => r.json());
        }

        if (cmd === 'scan') {
            return fetch(`${base}/api/scan`, { method: 'POST' }).then(r => r.json());
        }

        if (cmd === 'trade') {
            return fetch(`${base}/api/trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            }).then(r => r.json());
        }

        // Generic control
        return fetch(`${base}/api/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: cmd, data: payload }),
        }).then(r => r.json());
    }

    disconnect(): void {
        this.socket?.disconnect();
        this.socket = null;
    }
}

export default WebSocketManager;
