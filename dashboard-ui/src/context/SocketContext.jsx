import React, { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const SocketContext = createContext();

export const useSocket = () => {
    return useContext(SocketContext);
};

export const SocketProvider = ({ children }) => {
    const [socket, setSocket] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [systemState, setSystemState] = useState(null);

    useEffect(() => {
        // Connect to Flask Backend
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
        const newSocket = io(API_URL, {
            transports: ['websocket'],
            reconnectionAttempts: 5,
        });

        // Fetch initial state on connection
        const fetchInitialState = async () => {
            try {
                const [hubRes, signalsRes] = await Promise.all([
                    fetch(`${API_URL}/api/hub/state`).then(r => r.json()),
                    fetch(`${API_URL}/api/signals`).then(r => r.json())
                ]);

                console.log('>> [Initial Fetch] Hub state keys:', Object.keys(hubRes));
                console.log('>> [Initial Fetch] Signals:', signalsRes.signals?.length);

                // Hub state is already flat from the aggregator — merge directly
                const radar = signalsRes.signals || hubRes.radar || [];

                setSystemState(prev => ({
                    ...prev,
                    ...hubRes,
                    radar,
                    live_signals: radar,
                    signals: signalsRes
                }));
            } catch (err) {
                console.error('>> [Initial Fetch] Error:', err);
            }
        };

        newSocket.on('connect', () => {
            console.log('>> [Socket] Connected');
            setIsConnected(true);
            fetchInitialState();
        });

        newSocket.on('disconnect', () => {
            console.log('>> [Socket] Disconnected');
            setIsConnected(false);
        });

        newSocket.on('state_update', (data) => {
            console.log('>> [Socket] state_update:', data);
            setSystemState(prev => ({ ...prev, ...data }));
        });

        newSocket.on('hub_state', (data) => {
            // New flat structure from backend - just merge it
            // console.log('>> [Socket] hub_state:', data.timestamp);
            setSystemState(prev => ({
                ...prev,
                ...data
            }));
        });

        newSocket.on('signals_update', (data) => {
            console.log('>> [Socket] signals_update:', data.signals?.length, 'signals');
            // Transform signals from generate_signal_report format to RadarPanel format
            const rawSignals = data.signals || [];
            const transformedSignals = rawSignals.map(sig => ({
                symbol: sig.symbol,
                // Unified Conviction: Prefer explicit conviction, fallback to score (assumed 0-1 if <1, else /100 ?)
                // Scout data usually has score 0.95. Real signals have conviction 0.xx.
                conviction: sig.conviction !== undefined ? sig.conviction : (sig.score || 0),

                score: sig.conviction ? sig.conviction * 100 : (sig.score ? sig.score * 100 : 0),

                quality: sig.quality || 'MEDIUM',
                status: sig.quality || 'MEDIUM',

                reason: sig.reason,
                direction: sig.direction || 'NEUTRAL',
                price: sig.price || 0,
                tp: sig.tp,
                sl: sig.sl,
                timestamp: sig.timestamp,
                regime: sig.regime,
                tda_status: sig.tda_status,
                metadata: sig.metadata,
                pips_potential: sig.pips_potential,
                expected_yield: sig.expected_yield,
                hit_probability: sig.hit_probability,
                decay_score: sig.decay_score,
                optimal_horizon: sig.optimal_horizon,
                execution_details: sig.execution_details
            }));

            setSystemState(prev => ({
                ...prev,
                signals: data,
                // Also update live_signals for RadarPanel which can merge with scout_data
                live_signals: transformedSignals
            }));
        });

        setSocket(newSocket);

        return () => {
            newSocket.disconnect();
        };
    }, []);

    const sendCommand = (cmd, payload = {}) => {
        const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
        return fetch(`${API_URL}/api/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: cmd, data: payload })
        }).then(res => res.json());
    };

    return (
        <SocketContext.Provider value={{ socket, isConnected, systemState, sendCommand }}>
            {children}
        </SocketContext.Provider>
    );
};
