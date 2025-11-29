import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Ensure mind directory exists
const mindDir = path.join(__dirname, 'public', 'mind');
if (!fs.existsSync(mindDir)) {
  fs.mkdirSync(mindDir, { recursive: true });
}

// API: Save memory image
app.post('/api/save-memory', async (req, res) => {
  try {
    const { imageData, label, prompt, timestamp } = req.body;

    // Create filename from timestamp and sanitized label
    const sanitizedLabel = (label || 'memory').replace(/[^a-z0-9]/gi, '_').substring(0, 30);
    const filename = `${timestamp || Date.now()}_${sanitizedLabel}.png`;
    const filepath = path.join(mindDir, filename);

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
      timestamp: timestamp || Date.now(),
      filename
    }, null, 2));

    console.log(`Saved memory: ${filename}`);

    res.json({ success: true, filename });
  } catch (error) {
    console.error('Failed to save memory:', error);
    res.status(500).json({ success: false, error: String(error) });
  }
});

// API: List all saved memories
app.get('/api/memories', (req, res) => {
  try {
    const files = fs.readdirSync(mindDir);
    const memories = [];

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

    res.json(memories);
  } catch (error) {
    console.error('Failed to list memories:', error);
    res.status(500).json({ error: String(error) });
  }
});

// API: Delete a memory
app.delete('/api/memories/:filename', (req, res) => {
  try {
    const { filename } = req.params;

    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const imagePath = path.join(mindDir, filename);
    const metaPath = imagePath.replace('.png', '.json');

    // Delete both files if they exist
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }
    if (fs.existsSync(metaPath)) {
      fs.unlinkSync(metaPath);
    }

    console.log(`Deleted memory: ${filename}`);
    res.json({ success: true });
  } catch (error) {
    console.error('Failed to delete memory:', error);
    res.status(500).json({ error: String(error) });
  }
});

// Serve static files from the dist folder (production build)
app.use(express.static(path.join(__dirname, 'dist')));

// Serve files from public folder (for mind images and other assets)
app.use(express.static(path.join(__dirname, 'public')));

// SPA fallback - serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Memory storage: ${mindDir}`);
});
