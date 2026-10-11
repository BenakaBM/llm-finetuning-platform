/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#0a0e14",
          900: "#10151d",
          850: "#141a24",
          800: "#1a2130",
          700: "#26303f",
          600: "#3a4658",
        },
        accent: {
          DEFAULT: "#4f8cff",
          dim: "#2f5cad",
          soft: "#1c2a42",
        },
        good: "#3ecf8e",
        warn: "#e8b84b",
        bad: "#f0685f",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular"],
      },
    },
  },
  plugins: [],
};
