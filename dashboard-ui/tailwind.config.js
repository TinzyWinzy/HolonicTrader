/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'bb-bg': '#0a0a0a',
                'bb-card': '#111111',
                'bb-elevated': '#1a1a1a',
                'bb-hover': '#1e1e1e',
                'bb-border': '#222222',
                'bb-accent': '#ffb300',
                'bb-positive': '#00e676',
                'bb-negative': '#ff1744',
                'bb-info': '#2979ff',
                'bb-warning': '#ff9100',
                'bb-arb': '#ab47bc',
            },
            fontFamily: {
                'mono': ['"JetBrains Mono"', '"Fira Code"', '"Cascadia Code"', 'monospace'],
                'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            },
            fontSize: {
                'xxs': '0.625rem',  // 10px
            },
            spacing: {
                '0.25': '1px',
                '0.75': '3px',
            },
        },
    },
    plugins: [],
}
