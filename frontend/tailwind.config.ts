import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        clinical: {
          bg: "#f7f9fb",
          surface: "#ffffff",
          border: "#e5e9ef",
          ink: "#0f1e2e",
          muted: "#5b6b7c",
          accent: "#0a6cff",
          positive: "#c1121f",
          negative: "#118a3c",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI",
          "Roboto", "Helvetica Neue", "Arial", "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
