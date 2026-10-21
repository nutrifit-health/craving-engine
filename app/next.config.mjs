import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDirectory = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['@heroui/react', '@heroui/styles', '@heroui-pro/react', '@nutrifit/web-ui'],
  outputFileTracingRoot: path.resolve(currentDirectory, '../../..'),
};

export default nextConfig;
