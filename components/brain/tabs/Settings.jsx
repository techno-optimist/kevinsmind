import { useState, useEffect, useCallback } from 'react';
import { useNivek } from '../BrainDashboard.jsx';

export default function Settings() {
  const { settings, setSettings, connectionStatus, memories, voiceSamples, conversations } = useNivek();
  const [showApiKey, setShowApiKey] = useState(false);
  const [testStatus, setTestStatus] = useState(null);

  // Server control state
  const [serverStatus, setServerStatus] = useState(null);
  const [serverLogs, setServerLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  // Fetch server status
  const fetchServerStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/status');
      if (response.ok) {
        const data = await response.json();
        setServerStatus(data);
      } else {
        setServerStatus({ status: 'error', message: 'Server returned error' });
      }
    } catch (err) {
      setServerStatus({ status: 'offline', message: 'Cannot connect to server' });
    }
  }, []);

  // Fetch server logs
  const fetchServerLogs = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/logs?limit=50');
      if (response.ok) {
        const data = await response.json();
        setServerLogs(data.logs || []);
      }
    } catch (err) {
      console.error('Failed to fetch logs:', err);
    }
  }, []);

  // Refresh both status and logs
  const refreshServerInfo = useCallback(async () => {
    setIsRefreshing(true);
    await Promise.all([fetchServerStatus(), fetchServerLogs()]);
    setIsRefreshing(false);
  }, [fetchServerStatus, fetchServerLogs]);

  // Auto-refresh server status every 5 seconds
  useEffect(() => {
    fetchServerStatus();
    const interval = setInterval(fetchServerStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchServerStatus]);

  // Fetch logs when panel is opened
  useEffect(() => {
    if (showLogs) {
      fetchServerLogs();
      const interval = setInterval(fetchServerLogs, 3000);
      return () => clearInterval(interval);
    }
  }, [showLogs, fetchServerLogs]);

  const testConnection = async () => {
    setTestStatus('testing');

    // Test WebSocket connection
    if (connectionStatus === 'connected') {
      setTestStatus('success');
      setTimeout(() => setTestStatus(null), 3000);
    } else {
      setTestStatus('error');
      setTimeout(() => setTestStatus(null), 3000);
    }
  };

  const clearAllData = () => {
    if (confirm('Are you sure you want to clear ALL data? This cannot be undone.')) {
      localStorage.clear();
      window.location.reload();
    }
  };

  const exportData = () => {
    const data = {
      identity: JSON.parse(localStorage.getItem('nivek_identity') || '{}'),
      memories: JSON.parse(localStorage.getItem('nivek_memories') || '[]'),
      voiceSamples: JSON.parse(localStorage.getItem('nivek_voice_samples') || '[]'),
      conversations: JSON.parse(localStorage.getItem('nivek_conversations') || '[]'),
      settings: JSON.parse(localStorage.getItem('nivek_settings') || '{}'),
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nivek-backup-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importData = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target.result);

        if (data.identity) localStorage.setItem('nivek_identity', JSON.stringify(data.identity));
        if (data.memories) localStorage.setItem('nivek_memories', JSON.stringify(data.memories));
        if (data.voiceSamples) localStorage.setItem('nivek_voice_samples', JSON.stringify(data.voiceSamples));
        if (data.conversations) localStorage.setItem('nivek_conversations', JSON.stringify(data.conversations));
        if (data.settings) localStorage.setItem('nivek_settings', JSON.stringify(data.settings));

        alert('Data imported successfully! Reloading...');
        window.location.reload();
      } catch (err) {
        alert('Failed to import data: ' + err.message);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Settings</h2>
        <p>Configure NIVEK's backend, LLM provider, and data management.</p>
      </div>

      <div className="tab-body">
        <div className="grid-2">
          {/* LLM Configuration */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <BrainIcon /> LLM Configuration
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">Provider</label>
                <select
                  className="form-select"
                  value={settings.llmProvider}
                  onChange={(e) => updateSetting('llmProvider', e.target.value)}
                >
                  <option value="claude">Anthropic Claude</option>
                  <option value="openai">OpenAI</option>
                  <option value="local">Local (Ollama)</option>
                  <option value="mock">Mock (No LLM)</option>
                </select>
                <p className="form-hint">
                  Select which LLM powers NIVEK's thinking.
                </p>
              </div>

              {settings.llmProvider !== 'mock' && settings.llmProvider !== 'local' && (
                <>
                  <div className="form-group">
                    <label className="form-label">Model</label>
                    <select
                      className="form-select"
                      value={settings.llmModel}
                      onChange={(e) => updateSetting('llmModel', e.target.value)}
                    >
                      {settings.llmProvider === 'claude' && (
                        <>
                          <option value="claude-sonnet-4-20250514">Claude Sonnet 4 (Recommended)</option>
                          <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet</option>
                          <option value="claude-3-5-haiku-20241022">Claude 3.5 Haiku (Fast)</option>
                        </>
                      )}
                      {settings.llmProvider === 'openai' && (
                        <>
                          <option value="gpt-4o">GPT-4o</option>
                          <option value="gpt-4o-mini">GPT-4o Mini (Fast)</option>
                        </>
                      )}
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="form-label">API Key</label>
                    <div className="api-key-input">
                      <input
                        type={showApiKey ? 'text' : 'password'}
                        className="form-input"
                        value={settings.apiKey}
                        onChange={(e) => updateSetting('apiKey', e.target.value)}
                        placeholder={settings.llmProvider === 'claude' ? 'sk-ant-...' : 'sk-...'}
                      />
                      <button
                        className="api-key-toggle"
                        onClick={() => setShowApiKey(!showApiKey)}
                        type="button"
                      >
                        {showApiKey ? <EyeOffIcon /> : <EyeIcon />}
                      </button>
                    </div>
                    <p className="form-hint">
                      Your API key is stored locally and never sent to our servers.
                    </p>
                  </div>
                </>
              )}

              {settings.llmProvider === 'local' && (
                <div className="form-group">
                  <label className="form-label">Ollama Model</label>
                  <input
                    type="text"
                    className="form-input"
                    value={settings.llmModel}
                    onChange={(e) => updateSetting('llmModel', e.target.value)}
                    placeholder="llama3.2"
                  />
                  <p className="form-hint">
                    Make sure Ollama is running locally on port 11434.
                  </p>
                </div>
              )}

              <button
                className="btn btn-secondary"
                onClick={testConnection}
                disabled={testStatus === 'testing'}
                style={{ marginTop: '0.5rem' }}
              >
                {testStatus === 'testing' ? 'Testing...' : 'Test Connection'}
              </button>

              {testStatus === 'success' && (
                <div className="status-success" style={{ marginTop: '1rem' }}>
                  <CheckIcon /> Backend connected successfully
                </div>
              )}
              {testStatus === 'error' && (
                <div className="status-error" style={{ marginTop: '1rem' }}>
                  <XIcon /> Connection failed. Is the backend running?
                </div>
              )}
            </div>

            {/* TTS Settings */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <VoiceIcon /> Voice Synthesis
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">TTS Model</label>
                <select
                  className="form-select"
                  value={settings.ttsModel}
                  onChange={(e) => updateSetting('ttsModel', e.target.value)}
                >
                  <option value="chatterbox">Chatterbox (Fast Voice Cloning)</option>
                  <option value="csm-1b">Sesame CSM-1B (Voice Cloning)</option>
                  <option value="mock">Mock Audio (Testing)</option>
                </select>
                <p className="form-hint">
                  {settings.ttsModel === 'chatterbox'
                    ? 'Chatterbox: Fast GPU-accelerated TTS with 5s voice cloning.'
                    : settings.ttsModel === 'csm-1b'
                    ? 'CSM-1B: High-quality voice cloning (~5-7s per sentence).'
                    : 'Mock: Instant sine wave for testing.'}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <input
                  type="checkbox"
                  id="ttsEnabled"
                  checked={settings.ttsEnabled}
                  onChange={(e) => updateSetting('ttsEnabled', e.target.checked)}
                />
                <label htmlFor="ttsEnabled" style={{ fontSize: '0.875rem' }}>
                  Enable voice responses
                </label>
              </div>
            </div>

            {/* Server Control */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <ServerIcon /> Server Control
                </span>
                <button
                  className="btn-icon"
                  onClick={refreshServerInfo}
                  disabled={isRefreshing}
                  title="Refresh"
                  style={{ marginLeft: 'auto' }}
                >
                  <RefreshIcon spinning={isRefreshing} />
                </button>
              </div>

              {/* Server Status */}
              <div className="metrics" style={{ marginBottom: '1rem' }}>
                <div className="metric">
                  <span className="metric-label">Status</span>
                  <span className="metric-value" style={{
                    color: serverStatus?.status === 'running' ? 'var(--success)' :
                           serverStatus?.status === 'offline' ? 'var(--error)' : 'var(--warning)'
                  }}>
                    {serverStatus?.status || 'checking...'}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Model</span>
                  <span className="metric-value">
                    {serverStatus?.model_loaded ? 'loaded' : 'not loaded'}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">Backend</span>
                  <span className="metric-value">{serverStatus?.backend || '-'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Uptime</span>
                  <span className="metric-value">{serverStatus?.uptime_formatted || '-'}</span>
                </div>
              </div>

              {serverStatus?.status === 'offline' && (
                <div className="status-error" style={{ marginBottom: '1rem' }}>
                  <XIcon /> Server is offline. Start it with: <code>python server_chatterbox.py</code>
                </div>
              )}

              {/* Logs Toggle */}
              <button
                className="btn btn-secondary"
                onClick={() => setShowLogs(!showLogs)}
                style={{ width: '100%', marginBottom: showLogs ? '1rem' : 0 }}
              >
                <LogIcon /> {showLogs ? 'Hide Logs' : 'Show Logs'}
              </button>

              {/* Logs Panel */}
              {showLogs && (
                <div className="logs-panel">
                  <div className="logs-container">
                    {serverLogs.length === 0 ? (
                      <div className="logs-empty">No logs available</div>
                    ) : (
                      serverLogs.map((log, i) => (
                        <div key={i} className={`log-entry log-${log.level.toLowerCase()}`}>
                          <span className="log-time">{log.time}</span>
                          <span className="log-level">{log.level}</span>
                          <span className="log-message">{log.message}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Data Management */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <DataIcon /> Data Storage
                </span>
              </div>

              <div className="metrics" style={{ marginBottom: '1rem' }}>
                <div className="metric">
                  <span className="metric-label">Memories</span>
                  <span className="metric-value">{memories?.length || 0}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Voice Samples</span>
                  <span className="metric-value">{voiceSamples?.length || 0}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Conversations</span>
                  <span className="metric-value">{conversations?.length || 0}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Storage</span>
                  <span className="metric-value">
                    {(new Blob([JSON.stringify(localStorage)]).size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>

              <p style={{ fontSize: '0.813rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                All data is stored locally in your browser. Export regularly for backup.
              </p>

              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button className="btn btn-secondary" onClick={exportData}>
                  <DownloadIcon /> Export Backup
                </button>
                <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
                  <UploadIcon /> Import Backup
                  <input
                    type="file"
                    accept=".json"
                    onChange={importData}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <WarnIcon /> Danger Zone
                </span>
              </div>

              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                These actions are irreversible. Make sure to export a backup first.
              </p>

              <button className="btn btn-danger" onClick={clearAllData}>
                <TrashIcon /> Clear All Data
              </button>
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <InfoIcon /> About NIVEK
                </span>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <p style={{ marginBottom: '0.5rem' }}>
                  <strong>Phase 0: The Latency Experiment</strong>
                </p>
                <p style={{ marginBottom: '0.75rem' }}>
                  Testing presence-bridging techniques in a web interface before moving to hardware.
                </p>
                <div className="metrics">
                  <div className="metric">
                    <span className="metric-label">Version</span>
                    <span className="metric-value">0.2.0</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">TTS Engine</span>
                    <span className="metric-value">Chatterbox</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Backend</span>
                    <span className="metric-value">FastAPI + WebSocket</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// === ICONS ===
function BrainIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a8 8 0 0 0-8 8c0 2.5 1.5 4.5 3 6s2 3 2 4h6c0-1 .5-2.5 2-4s3-3.5 3-6a8 8 0 0 0-8-8z"/>
      <circle cx="12" cy="10" r="2"/>
    </svg>
  );
}

function VoiceIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    </svg>
  );
}

function DataIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  );
}

function EyeOffIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>
  );
}

function WarnIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>
  );
}

function ServerIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
      <line x1="6" y1="6" x2="6.01" y2="6"/>
      <line x1="6" y1="18" x2="6.01" y2="18"/>
    </svg>
  );
}

function RefreshIcon({ spinning }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      style={spinning ? { animation: 'spin 1s linear infinite' } : undefined}
    >
      <polyline points="23 4 23 10 17 10"/>
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
    </svg>
  );
}

function LogIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>
  );
}
