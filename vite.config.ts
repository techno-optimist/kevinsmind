import path from 'path';
import fs from 'fs';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Custom plugin to handle saving memory images and serve books
function memoryStoragePlugin() {
  return {
    name: 'memory-storage',
    configureServer(server: any) {
      // Serve PDF books from the books folder
      server.middlewares.use('/books', (req: any, res: any, next: any) => {
        const booksDir = path.join(process.cwd(), 'books');
        const requestedFile = decodeURIComponent(req.url.replace(/^\//, ''));
        const filePath = path.join(booksDir, requestedFile);

        // Security: prevent directory traversal
        if (!filePath.startsWith(booksDir)) {
          res.writeHead(403);
          res.end('Forbidden');
          return;
        }

        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          const ext = path.extname(filePath).toLowerCase();
          const contentType = ext === '.pdf' ? 'application/pdf' : 'application/octet-stream';
          res.writeHead(200, { 'Content-Type': contentType });
          fs.createReadStream(filePath).pipe(res);
        } else {
          next();
        }
      });

      // POST /api/save-memory - Save a generated image to the mind folder
      server.middlewares.use('/api/save-memory', async (req: any, res: any, next: any) => {
        if (req.method !== 'POST') {
          return next();
        }

        let body = '';
        req.on('data', (chunk: any) => { body += chunk.toString(); });
        req.on('end', async () => {
          try {
            const { imageData, label, prompt, comment, userInput, timestamp } = JSON.parse(body);

            // Create filename from timestamp and sanitized label
            const sanitizedLabel = (label || 'memory').replace(/[^a-z0-9]/gi, '_').substring(0, 30);
            const filename = `${timestamp || Date.now()}_${sanitizedLabel}.png`;
            const filepath = path.join(process.cwd(), 'public', 'mind', filename);

            // Extract base64 data (remove data URL prefix if present)
            const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
            const buffer = Buffer.from(base64Data, 'base64');

            // Save the image
            fs.writeFileSync(filepath, buffer);

            // Save metadata alongside
            const metaPath = filepath.replace('.png', '.json');
            fs.writeFileSync(metaPath, JSON.stringify({
              label,
              prompt,
              userInput: userInput || '',
              comment: comment || '',
              timestamp: timestamp || Date.now(),
              filename
            }, null, 2));

            console.log(`Saved memory: ${filename}`);

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, filename }));
          } catch (error) {
            console.error('Failed to save memory:', error);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: String(error) }));
          }
        });
      });

      // GET /api/memories - List all saved memories
      server.middlewares.use('/api/memories', (req: any, res: any, next: any) => {
        if (req.method !== 'GET') {
          return next();
        }

        try {
          const mindDir = path.join(process.cwd(), 'public', 'mind');

          // Ensure directory exists
          if (!fs.existsSync(mindDir)) {
            fs.mkdirSync(mindDir, { recursive: true });
          }

          const files = fs.readdirSync(mindDir);
          const memories: any[] = [];

          // Load all .json metadata files
          for (const file of files) {
            if (file.endsWith('.json')) {
              const metaPath = path.join(mindDir, file);
              const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
              const imagePath = `/mind/${meta.filename}`;
              memories.push({
                ...meta,
                imagePath
              });
            }
          }

          // Sort by timestamp (newest first)
          memories.sort((a, b) => b.timestamp - a.timestamp);

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(memories));
        } catch (error) {
          console.error('Failed to list memories:', error);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: String(error) }));
        }
      });
    }
  };
}

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react(), memoryStoragePlugin()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
