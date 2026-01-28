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

        newSocket.on('connect', () => {
            console.log('>> [Socket] Connected');
            setIsConnected(true);
        });

        newSocket.on('disconnect', () => {
            console.log('>> [Socket] Disconnected');
            setIsConnected(false);
        });

        newSocket.on('state_update', (data) => {
            // console.log('>> [Socket] State Update:', data);
            setSystemState(data);
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
