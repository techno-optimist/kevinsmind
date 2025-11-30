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
    const { imageData, label, prompt, comment, userInput, timestamp } = req.body;

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
      userInput: userInput || '',
      comment: comment || '',
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

// API: Export all memories as a single JSON bundle (for backup)
app.get('/api/memories/export', async (req, res) => {
  try {
    const files = fs.readdirSync(mindDir);
    const memoriesBundle = [];

    for (const file of files) {
      if (file.endsWith('.json')) {
        const metaPath = path.join(mindDir, file);
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));

        // Read the corresponding image and include as base64
        const imagePath = path.join(mindDir, meta.filename);
        if (fs.existsSync(imagePath)) {
          const imageBuffer = fs.readFileSync(imagePath);
          const imageBase64 = imageBuffer.toString('base64');
          memoriesBundle.push({
            ...meta,
            imageData: `data:image/png;base64,${imageBase64}`
          });
        }
      }
    }

    // Sort by timestamp
    memoriesBundle.sort((a, b) => a.timestamp - b.timestamp);

    console.log(`Exported ${memoriesBundle.length} memories`);

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', `attachment; filename="kevinsmind-backup-${Date.now()}.json"`);
    res.json({
      exportedAt: Date.now(),
      version: 1,
      count: memoriesBundle.length,
      memories: memoriesBundle
    });
  } catch (error) {
    console.error('Failed to export memories:', error);
    res.status(500).json({ error: String(error) });
  }
});

// API: Import memories from a backup bundle
app.post('/api/memories/import', async (req, res) => {
  try {
    const { memories, skipExisting } = req.body;

    if (!memories || !Array.isArray(memories)) {
      return res.status(400).json({ error: 'Invalid backup format: memories array required' });
    }

    let imported = 0;
    let skipped = 0;
    const errors = [];

    for (const memory of memories) {
      try {
        const { imageData, label, prompt, comment, userInput, timestamp, filename } = memory;

        if (!imageData || !timestamp) {
          errors.push(`Skipped invalid memory: missing imageData or timestamp`);
          continue;
        }

        // Use original filename or generate new one
        const sanitizedLabel = (label || 'memory').replace(/[^a-z0-9]/gi, '_').substring(0, 30);
        const newFilename = filename || `${timestamp}_${sanitizedLabel}.png`;
        const filepath = path.join(mindDir, newFilename);

        // Skip if file already exists and skipExisting is true
        if (skipExisting && fs.existsSync(filepath)) {
          skipped++;
          continue;
        }

        // Extract base64 data and save image
        const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
        const buffer = Buffer.from(base64Data, 'base64');
        fs.writeFileSync(filepath, buffer);

        // Save metadata
        const metaPath = filepath.replace('.png', '.json');
        fs.writeFileSync(metaPath, JSON.stringify({
          label,
          prompt,
          userInput: userInput || '',
          comment: comment || '',
          timestamp,
          filename: newFilename
        }, null, 2));

        imported++;
      } catch (memError) {
        errors.push(`Failed to import memory: ${String(memError)}`);
      }
    }

    console.log(`Import complete: ${imported} imported, ${skipped} skipped, ${errors.length} errors`);
    res.json({
      success: true,
      imported,
      skipped,
      errors: errors.length > 0 ? errors : undefined
    });
  } catch (error) {
    console.error('Failed to import memories:', error);
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

  // Log disk/storage status on startup
  try {
    const files = fs.readdirSync(mindDir);
    const jsonFiles = files.filter(f => f.endsWith('.json'));
    const pngFiles = files.filter(f => f.endsWith('.png'));
    console.log(`📦 Storage status: ${jsonFiles.length} memories found (${pngFiles.length} images)`);

    if (jsonFiles.length > 0) {
      console.log(`📝 Most recent memories:`);
      const memories = jsonFiles
        .map(f => {
          try {
            const meta = JSON.parse(fs.readFileSync(path.join(mindDir, f), 'utf-8'));
            return { label: meta.label, timestamp: meta.timestamp };
          } catch { return null; }
        })
        .filter(Boolean)
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 3);
      memories.forEach(m => console.log(`   - ${m.label} (${new Date(m.timestamp).toISOString()})`));
    } else {
      console.log(`⚠️  No memories found in storage directory`);
    }
  } catch (err) {
    console.error(`❌ Error reading storage directory:`, err.message);
  }
});
