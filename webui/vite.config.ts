import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bridge is on 8080 so that 8000 stays free for the SSH tunnel to the GPU
// node (`ssh -N -L 8000:localhost:8000 <node>`). Proxying /api here is also why
// webapi/bridge_app.py ships no CORS middleware: the browser is same-origin.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
    },
  },
});
