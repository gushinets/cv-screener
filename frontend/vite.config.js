import {defineConfig} from 'vite';
export default defineConfig({
  server: {proxy: {'/api': {target: 'http://127.0.0.1:7860', changeOrigin: true,
    configure: proxy => proxy.on('proxyReq', request => request.setHeader('origin', 'http://127.0.0.1:7860'))}}},
  build: {target: 'es2022'},
});
