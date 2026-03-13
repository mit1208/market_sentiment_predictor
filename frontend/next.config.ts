import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // google-auth-library uses native Node.js crypto/tls modules that webpack can't bundle.
  // Marking it external ensures Next.js loads it directly at runtime instead.
  serverExternalPackages: ["google-auth-library"],
};

export default nextConfig;
