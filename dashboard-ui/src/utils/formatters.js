/**
 * Formats a price with dynamic precision based on its value.
 * - < 0.01: 8 decimals (e.g. 0.00001234)
 * - < 1.00: 6 decimals (e.g. 0.123456)
 * - >= 1.00: 2 decimals (e.g. 1234.56)
 * @param {number|string} price - The price to format
 * @returns {string} Formatted price string
 */
export const formatPrice = (price) => {
    const p = parseFloat(price);
    if (isNaN(p)) return '0.00';
    if (p === 0) return '0.00';

    if (p < 0.01) return p.toFixed(8);
    if (p < 1.0) return p.toFixed(6);
    return p.toFixed(2);
};
