import React from 'react';

const DashboardLayout = ({ children }) => {
    return (
        <div className="min-h-screen h-screen bg-holon-bg text-holon-text flex flex-col overflow-hidden">
            {/* Header - Fixed to Top */}
            <header className="flex-shrink-0 flex justify-between items-center border-b border-holon-card bg-holon-bg/95 backdrop-blur z-10 px-4 py-3 md:px-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-holon-accent to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-emerald-900/20">
                        <span className="font-orbitron font-bold text-white text-lg">H</span>
                    </div>
                    <div>
                        <h1 className="text-xl font-orbitron tracking-wide font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-holon-dim">
                            HOLONIC TRADER <span className="text-xs text-holon-accent align-top">v2.1</span>
                        </h1>
                        <p className="text-xs text-holon-dim uppercase tracking-widest font-mono">NEXUS COMMAND CENTER</p>
                    </div>
                </div>
                <div id="header-actions">
                    {/* Placeholder for Actions */}
                </div>
            </header>

            {/* Main Content - Scrollable */}
            <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 md:p-6">
                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 max-w-[1920px] mx-auto items-start">
                    {children}
                </div>
            </main>
        </div>
    );
};

export default DashboardLayout;
