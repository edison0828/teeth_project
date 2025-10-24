import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0F172A",
        surface: "#111B2E",
        primary: {
          DEFAULT: "#3CC4FF",
          subtle: "#9BDFFF"
        },
        accent: "#26E1A6",
        warning: "#F7B955",
        danger: "#FF6B6B"
      },
      boxShadow: {
        card: "0 20px 45px rgba(15, 23, 42, 0.35)"
      }
    }
  },
  plugins: []
};

export default config;
