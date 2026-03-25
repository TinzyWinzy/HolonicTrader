import React, { useEffect, useRef, useState } from 'react';

interface DataCellProps {
    value: number | string;
    format?: 'price' | 'pct' | 'pnl' | 'usd' | 'raw';
    decimals?: number;
    flashOnChange?: boolean;
    className?: string;
}

const DataCell: React.FC<DataCellProps> = ({
    value,
    format = 'raw',
    decimals = 2,
    flashOnChange = true,
    className = '',
}) => {
    const prevValue = useRef(value);
    const [flashClass, setFlashClass] = useState('');

    useEffect(() => {
        if (!flashOnChange) return;
        const prev = typeof prevValue.current === 'number' ? prevValue.current : parseFloat(String(prevValue.current));
        const curr = typeof value === 'number' ? value : parseFloat(String(value));

        if (!isNaN(prev) && !isNaN(curr) && prev !== curr) {
            setFlashClass(curr > prev ? 'flash-positive' : 'flash-negative');
            const t = setTimeout(() => setFlashClass(''), 600);
            return () => clearTimeout(t);
        }
        prevValue.current = value;
    }, [value, flashOnChange]);

    const numValue = typeof value === 'number' ? value : parseFloat(String(value));
    const isPositive = numValue >= 0;

    let display: string;
    let colorClass: string;

    switch (format) {
        case 'price':
            display = isNaN(numValue) ? '—' : numValue.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
            colorClass = 'text-[var(--text-primary)]';
            break;
        case 'pct':
            display = isNaN(numValue) ? '—' : `${isPositive ? '+' : ''}${numValue.toFixed(decimals)}%`;
            colorClass = isPositive ? 'text-[var(--color-positive)]' : 'text-[var(--color-negative)]';
            break;
        case 'pnl':
            display = isNaN(numValue) ? '—' : `${isPositive ? '+' : ''}$${numValue.toFixed(decimals)}`;
            colorClass = isPositive ? 'text-[var(--color-positive)]' : 'text-[var(--color-negative)]';
            break;
        case 'usd':
            display = isNaN(numValue) ? '—' : `$${numValue.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
            colorClass = 'text-[var(--text-primary)]';
            break;
        default:
            display = String(value);
            colorClass = 'text-[var(--text-primary)]';
    }

    return (
        <span
            className={`font-mono text-[var(--text-sm)] ${colorClass} ${flashClass} ${className}`}
            style={{ fontFamily: 'var(--font-mono)' }}
        >
            {display}
        </span>
    );
};

export default DataCell;
