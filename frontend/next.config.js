/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export — FastAPI serves the resulting frontend/out/ directory as
  // the site root, and the frontend calls the backend via relative /api/* URLs.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
