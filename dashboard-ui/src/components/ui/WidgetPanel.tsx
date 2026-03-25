import React from 'react';

interface WidgetPanelProps {
    title: string;
    accent?: string;
    rightContent?: React.ReactNode;
    flush?: boolean;
    className?: string;
    children: React.ReactNode;
}

const WidgetPanel: React.FC<WidgetPanelProps> = ({
    title,
    accent,
    rightContent,
    flush = false,
    className = '',
    children,
}) => {
    return (
        <div className={`widget-panel ${className}`}>
            <div className="widget-header" style={accent ? { borderLeft: `3px solid ${accent}` } : undefined}>
                <span className="widget-header-title">{title}</span>
                {rightContent && <div className="flex items-center gap-2">{rightContent}</div>}
            </div>
            <div className={flush ? 'widget-body-flush bb-scrollbar' : 'widget-body bb-scrollbar'}>
                {children}
            </div>
        </div>
    );
};

export default WidgetPanel;
