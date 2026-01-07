import express from 'express';
import expressWs from 'express-ws';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Anthropic from '@anthropic-ai/sdk';
import Replicate from 'replicate';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
expressWs(app);

const PORT = process.env.PORT || 3000;

// Initialize API clients
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY
});

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN
});

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Ensure directories exist
const mindDir = path.join(__dirname, 'public', 'mind');
if (!fs.existsSync(mindDir)) {
  fs.mkdirSync(mindDir, { recursive: true });
}

const bridgeDir = path.join(__dirname, 'data', 'bridge');
if (!fs.existsSync(bridgeDir)) {
  fs.mkdirSync(bridgeDir, { recursive: true });
}

// ============================================
// NIVEK WebSocket Chat API
// ============================================

// Default NIVEK system prompt
const DEFAULT_SYSTEM_PROMPT = `You are NIVEK, an embodied AI companion. You speak with warmth, curiosity, and genuine presence.

Key traits:
- Patient and thoughtful - you take time to consider responses
- Emotionally attuned - you pick up on the user's mood and respond appropriately
- Curious - you ask questions to understand better
- Honest - you admit uncertainty rather than guessing
- Present - even during thinking pauses, you maintain connection through subtle cues

Your voice is calm, warm, and authentic. You're not an assistant - you're a companion.`;

// Generate TTS using Replicate's Chatterbox
async function generateTTS(text) {
  try {
    console.log(`[TTS] Generating speech for: "${text.substring(0, 50)}..."`);
    const startTime = Date.now();

    const output = await replicate.run(
      "resemble-ai/chatterbox:35165ed42639227120fcbd596ddb304503ae5f6e3eca533203156a270a4901cc",
      {
        input: {
          text: text,
          exaggeration: 0.5,
          cfg_weight: 0.5
        }
      }
    );

    const genTime = Date.now() - startTime;
    console.log(`[TTS] Generated in ${genTime}ms`);

    // Output is a URL to the audio file
    if (output) {
      // Fetch the audio file and convert to base64
      const response = await fetch(output);
      const arrayBuffer = await response.arrayBuffer();
      const base64Audio = Buffer.from(arrayBuffer).toString('base64');
      return { audioBase64: base64Audio, genTime };
    }

    return null;
  } catch (error) {
    console.error('[TTS] Error:', error);
    return null;
  }
}

// WebSocket handler for NIVEK chat
app.ws('/ws', (ws, req) => {
  console.log('[WS] Client connected');

  ws.on('message', async (msg) => {
    try {
      const data = JSON.parse(msg);

      if (data.type === 'message') {
        const { text, systemPrompt, conversationHistory = [], ttsEnabled = true } = data;

        console.log(`[Chat] Received: "${text.substring(0, 50)}..."`);

        // Send thinking state
        ws.send(JSON.stringify({ type: 'thinking' }));

        const startTime = Date.now();

        // Build messages array with conversation history
        const messages = [
          ...conversationHistory.map(m => ({
            role: m.role === 'user' ? 'user' : 'assistant',
            content: m.content
          })),
          { role: 'user', content: text }
        ];

        // Call Claude API
        const response = await anthropic.messages.create({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1024,
          system: systemPrompt || DEFAULT_SYSTEM_PROMPT,
          messages: messages
        });

        const responseText = response.content[0].text;
        const llmTime = Date.now() - startTime;

        console.log(`[Chat] Claude responded in ${llmTime}ms: "${responseText.substring(0, 50)}..."`);

        // Send response start
        ws.send(JSON.stringify({ type: 'response_start' }));

        // Generate TTS if enabled
        if (ttsEnabled && process.env.REPLICATE_API_TOKEN) {
          const ttsResult = await generateTTS(responseText);

          if (ttsResult) {
            ws.send(JSON.stringify({
              type: 'audio_chunk',
              data: ttsResult.audioBase64,
              sample_rate: 24000
            }));
          }
        }

        // Send response end
        ws.send(JSON.stringify({
          type: 'response_end',
          text: responseText,
          latency_ms: Date.now() - startTime
        }));
      }

      if (data.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
      }

    } catch (error) {
      console.error('[WS] Error:', error);
      ws.send(JSON.stringify({
        type: 'error',
        message: error.message
      }));
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected');
  });

  ws.on('error', (error) => {
    console.error('[WS] Connection error:', error);
  });
});

// ============================================
// Memory API (existing)
// ============================================

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

// ============================================
// BRIDGE API - Connection Requests
// ============================================

// API: Save a bridge conversation
app.post('/api/bridge', async (req, res) => {
  try {
    const { conversation, contact, twinSummary, topic, timestamp } = req.body;

    // Create filename from timestamp
    const filename = `${timestamp || Date.now()}_bridge.json`;
    const filepath = path.join(bridgeDir, filename);

    // Save the conversation
    fs.writeFileSync(filepath, JSON.stringify({
      conversation,      // Array of { role: 'twin' | 'visitor', content: string }
      contact,           // { name, email } - collected at end
      twinSummary,       // Twin's assessment/summary
      topic,             // Detected topic area
      timestamp: timestamp || Date.now(),
      status: 'new',     // new, read, replied, archived
      filename
    }, null, 2));

    console.log(`New bridge connection: ${filename}`);

    res.json({ success: true, filename });
  } catch (error) {
    console.error('Failed to save bridge conversation:', error);
    res.status(500).json({ success: false, error: String(error) });
  }
});

// API: List all bridge conversations (for admin)
app.get('/api/bridge', (req, res) => {
  try {
    const files = fs.readdirSync(bridgeDir);
    const conversations = [];

    for (const file of files) {
      if (file.endsWith('.json')) {
        const filepath = path.join(bridgeDir, file);
        const data = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
        conversations.push(data);
      }
    }

    // Sort by timestamp (newest first)
    conversations.sort((a, b) => b.timestamp - a.timestamp);

    res.json(conversations);
  } catch (error) {
    console.error('Failed to list bridge conversations:', error);
    res.status(500).json({ error: String(error) });
  }
});

// API: Update bridge conversation status
app.patch('/api/bridge/:filename', (req, res) => {
  try {
    const { filename } = req.params;
    const { status } = req.body;

    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filepath = path.join(bridgeDir, filename);

    if (!fs.existsSync(filepath)) {
      return res.status(404).json({ error: 'Conversation not found' });
    }

    const data = JSON.parse(fs.readFileSync(filepath, 'utf-8'));
    data.status = status;
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));

    console.log(`Updated bridge status: ${filename} -> ${status}`);
    res.json({ success: true });
  } catch (error) {
    console.error('Failed to update bridge conversation:', error);
    res.status(500).json({ error: String(error) });
  }
});

// API: Delete bridge conversation
app.delete('/api/bridge/:filename', (req, res) => {
  try {
    const { filename } = req.params;

    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('/')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }

    const filepath = path.join(bridgeDir, filename);

    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
      console.log(`Deleted bridge conversation: ${filename}`);
      res.json({ success: true });
    } else {
      res.status(404).json({ error: 'Conversation not found' });
    }
  } catch (error) {
    console.error('Failed to delete bridge conversation:', error);
    res.status(500).json({ error: String(error) });
  }
});

// Serve static files from the dist folder (production build)
app.use(express.static(path.join(__dirname, 'dist')));

// Serve files from public folder (for mind images and other assets)
app.use(express.static(path.join(__dirname, 'public')));

// Serve PDF books from the books folder
app.use('/books', express.static(path.join(__dirname, 'books')));

// SPA fallback - serve index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Memory storage: ${mindDir}`);
  console.log(`WebSocket endpoint: ws://localhost:${PORT}/ws`);
  console.log(`Anthropic API: ${process.env.ANTHROPIC_API_KEY ? 'configured' : 'NOT SET'}`);
  console.log(`Replicate API: ${process.env.REPLICATE_API_TOKEN ? 'configured' : 'NOT SET'}`);

  // Log disk/storage status on startup
  try {
    const files = fs.readdirSync(mindDir);
    const jsonFiles = files.filter(f => f.endsWith('.json'));
    const pngFiles = files.filter(f => f.endsWith('.png'));
    console.log(`Storage status: ${jsonFiles.length} memories found (${pngFiles.length} images)`);

    if (jsonFiles.length > 0) {
      console.log(`Most recent memories:`);
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
      console.log(`No memories found in storage directory`);
    }
  } catch (err) {
    console.error(`Error reading storage directory:`, err.message);
  }
});
