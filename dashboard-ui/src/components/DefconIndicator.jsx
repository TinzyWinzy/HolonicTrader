import React from 'react';

/**
 * DefconIndicator - Crisis threat level display
 * Shows DEFCON 1-5 status with visual indicators
 */
const DefconIndicator = ({ doomsday = {} }) => {
    const level = doomsday.defcon_level || 5;
    const crisisActive = doomsday.crisis_active || false;

    const getDefconStyle = (defconLevel) => {
        switch (defconLevel) {
            case 1:
                return {
                    bg: 'bg-red-500',
                    border: 'border-red-400',
                    text: 'text-white',
                    glow: 'shadow-red-500/50',
                    label: 'CATASTROPHIC',
                    icon: '☢️'
                };
            case 2:
                return {
                    bg: 'bg-orange-500',
                    border: 'border-orange-400',
                    text: 'text-white',
                    glow: 'shadow-orange-500/50',
                    label: 'SEVERE',
                    icon: '🔴'
                };
            case 3:
                return {
                    bg: 'bg-yellow-500',
                    border: 'border-yellow-400',
                    text: 'text-black',
                    glow: 'shadow-yellow-500/30',
                    label: 'HIGH',
                    icon: '🟡'
                };
            case 4:
                return {
                    bg: 'bg-blue-500',
                    border: 'border-blue-400',
                    text: 'text-white',
                    glow: '',
                    label: 'ELEVATED',
                    icon: '🔵'
                };
            case 5:
            default:
                return {
                    bg: 'bg-emerald-500',
                    border: 'border-emerald-400',
                    text: 'text-white',
                    glow: '',
                    label: 'NORMAL',
                    icon: '🟢'
                };
        }
    };

    const style = getDefconStyle(level);

    return (
        <div className={`
            relative p-3 rounded-lg border-2 
            ${style.border} ${style.glow}
            bg-gray-900/80 backdrop-blur-sm
            ${crisisActive ? 'animate-pulse' : ''}
        `}>
            <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                    <span className="text-2xl">{style.icon}</span>
                    <div>
                        <p className="text-[10px] uppercase text-gray-400 font-mono tracking-wider">
                            Threat Level
                        </p>
                        <p className={`text-lg font-bold ${style.bg === 'bg-emerald-500' ? 'text-emerald-400' :
                            style.bg === 'bg-yellow-500' ? 'text-yellow-400' :
                                style.bg === 'bg-orange-500' ? 'text-orange-400' :
                                    style.bg === 'bg-red-500' ? 'text-red-400' : 'text-blue-400'
                            }`}>
                            DEFCON {level}
                        </p>
                    </div>
                </div>

                <div className={`
                    px-2 py-1 rounded text-xs font-bold uppercase
                    ${style.bg} ${style.text}
                `}>
                    {style.label}
                </div>
            </div>

            {/* Crisis actions taken */}
            {doomsday.actions_taken?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-700">
                    <p className="text-xs text-gray-500">
                        Actions: {doomsday.actions_taken.join(', ')}
                    </p>
                </div>
            )}

            {/* Safe haven indicator */}
            {crisisActive && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping" />
            )}
        </div>
    );
};

export default DefconIndicator;
