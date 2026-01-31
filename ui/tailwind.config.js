/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        'genesis-cyan': '#00CCFF',
        'genesis-blue': '#0080FF',
        'genesis-orange': '#FF8000',
        'genesis-gold': '#FFD700',
        'genesis-void': '#000033'
      },
      backdropBlur: {
        'glass': '20px'
      }
    }
  },
  plugins: []
};

