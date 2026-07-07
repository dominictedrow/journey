/** @type {import('next').NextConfig} */

// Two build targets from one codebase:
// - default (Docker): `output: "standalone"` — a small Node server (docker/frontend.Dockerfile).
// - desktop (Tauri):  `output: "export"` — a static bundle in `out/` that the `.dmg` ships and
//   serves from the webview; all data comes from the configurable backend URL at runtime.
// Set BUILD_TARGET=desktop (scripts/build-dmg.sh does) to produce the static export.
const desktop = process.env.BUILD_TARGET === "desktop";

const nextConfig = desktop
  ? {
      output: "export",
      // Emit `route/index.html` so file-based loading in the webview resolves clean paths.
      trailingSlash: true,
      // No Next image optimization server exists in a static bundle.
      images: { unoptimized: true },
    }
  : {
      output: "standalone",
    };

module.exports = nextConfig;
