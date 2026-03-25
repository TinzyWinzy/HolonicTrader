
import React, { useEffect, useState } from 'react';
import { useSocket } from '../context/SocketContext';
import { Bell, X } from 'lucide-react';
import clsx from 'clsx';

export default function SystemAlerts() {
    const { lastMessage } = useSocket();
    const [alerts, setAlerts] = useState([]);

    useEffect(() => {
        if (lastMessage && lastMessage.type === 'system_alert') {
            const newAlert = {
                id: Date.now(),
                ...lastMessage.data,
                timestamp: new Date().toLocaleTimeString()
            };
            setAlerts(prev => [newAlert, ...prev].slice(0, 3)); // Keep max 3

            // Auto dismiss
            setTimeout(() => {
                setAlerts(prev => prev.filter(a => a.id !== newAlert.id));
            }, 10000); // 10s
        }
    }, [lastMessage]);

    if (alerts.length === 0) return null;

    return (
        <div className="fixed top-4 right-4 left-4 md:left-auto md:w-96 z-50 flex flex-col gap-2">
            {alerts.map(alert => (
                <div
                    key={alert.id}
                    className={clsx(
                        "p-4 rounded-lg shadow-2xl border flex items-start gap-3 backdrop-blur-md animate-slide-in",
                        alert.level === 'POSITIVE' ? "bg-emerald-900/90 border-emerald-500/50 text-emerald-100" :
                            alert.level === 'CRITICAL' ? "bg-red-900/90 border-red-500/50 text-red-100" :
                                "bg-blue-900/90 border-blue-500/50 text-blue-100"
                    )}
                >
                    <Bell size={20} className="mt-1 shrink-0" />
                    <div className="flex-1">
                        <h4 className="font-bold text-sm mb-1">SYSTEM ALERT</h4>
                        <p className="text-sm font-mono">{alert.message}</p>
                        <p className="text-[10px] opacity-70 mt-2">{alert.timestamp}</p>
                    </div>
                    <button
                        onClick={() => setAlerts(prev => prev.filter(a => a.id !== alert.id))}
                        className="opacity-70 hover:opacity-100"
                    >
                        <X size={16} />
                    </button>
                </div>
            ))}
        </div>
    );
}
