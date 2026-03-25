import React from 'react';

type BadgeVariant = 'long' | 'short' | 'arb' | 'dir' | 'high' | 'medium' | 'low' | 'defcon' | 'info' | 'warning' | 'danger' | 'success';

interface BadgeProps {
    variant: BadgeVariant;
    children: React.ReactNode;
    className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
    long: 'bg-[rgba(0,230,118,0.15)] text-[var(--color-long)] border-[rgba(0,230,118,0.3)]',
    short: 'bg-[rgba(255,23,68,0.15)] text-[var(--color-short)] border-[rgba(255,23,68,0.3)]',
    arb: 'bg-[rgba(171,71,188,0.15)] text-[var(--color-arb)] border-[rgba(171,71,188,0.3)]',
    dir: 'bg-[rgba(41,121,255,0.15)] text-[var(--color-dir)] border-[rgba(41,121,255,0.3)]',
    high: 'bg-[rgba(0,230,118,0.15)] text-[var(--color-positive)] border-[rgba(0,230,118,0.3)]',
    medium: 'bg-[rgba(255,179,0,0.15)] text-[var(--accent-crypto)] border-[rgba(255,179,0,0.3)]',
    low: 'bg-[rgba(102,102,102,0.15)] text-[var(--text-muted)] border-[rgba(102,102,102,0.3)]',
    defcon: 'bg-[rgba(255,23,68,0.15)] text-[var(--color-negative)] border-[rgba(255,23,68,0.3)]',
    info: 'bg-[rgba(41,121,255,0.15)] text-[var(--color-info)] border-[rgba(41,121,255,0.3)]',
    warning: 'bg-[rgba(255,145,0,0.15)] text-[var(--color-warning)] border-[rgba(255,145,0,0.3)]',
    danger: 'bg-[rgba(255,23,68,0.15)] text-[var(--color-negative)] border-[rgba(255,23,68,0.3)]',
    success: 'bg-[rgba(0,230,118,0.15)] text-[var(--color-positive)] border-[rgba(0,230,118,0.3)]',
};

const Badge: React.FC<BadgeProps> = ({ variant, children, className = '' }) => {
    return (
        <span
            className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono font-semibold uppercase tracking-wider border rounded ${variantStyles[variant]} ${className}`}
        >
            {children}
        </span>
    );
};

export default Badge;
