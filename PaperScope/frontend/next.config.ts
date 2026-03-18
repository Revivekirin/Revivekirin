import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/Revivekirin',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
