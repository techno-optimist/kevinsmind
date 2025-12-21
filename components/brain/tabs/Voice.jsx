import { useState, useRef, useCallback } from 'react';
import { useNivek } from '../BrainDashboard.jsx';

export default function Voice() {
  const { voiceSamples, addVoiceSample, deleteVoiceSample, connectionStatus } = useNivek();

  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [sampleText, setSampleText] = useState('');

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 24000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000
      });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      audioChunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const channelData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(channelData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      mediaRecorderRef.current = { stream, audioContext, source, processor };
      setIsRecording(true);
      setRecordingDuration(0);

      // Timer for duration
      timerRef.current = setInterval(() => {
        setRecordingDuration(d => d + 1);
      }, 1000);

    } catch (err) {
      console.error('Failed to start recording:', err);
      alert('Could not access microphone. Please allow microphone access.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (!mediaRecorderRef.current || !isRecording) return;

    clearInterval(timerRef.current);

    const { stream, audioContext, source, processor } = mediaRecorderRef.current;

    source.disconnect();
    processor.disconnect();
    stream.getTracks().forEach(track => track.stop());
    audioContext.close();

    // Combine all audio chunks
    const totalLength = audioChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
    const audioData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunksRef.current) {
      audioData.set(chunk, offset);
      offset += chunk.length;
    }

    // Convert to base64
    const bytes = new Uint8Array(audioData.buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64Audio = btoa(binary);

    // Add sample
    addVoiceSample({
      text: sampleText || `Voice Sample ${voiceSamples.length + 1}`,
      audio: base64Audio,
      format: 'float32',
      sampleRate: 24000,
      duration: (totalLength / 24000).toFixed(1)
    });

    setSampleText('');
    setIsRecording(false);
    setRecordingDuration(0);
    mediaRecorderRef.current = null;
  }, [isRecording, sampleText, voiceSamples.length, addVoiceSample]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Voice Studio</h2>
        <p>Record voice samples to train NIVEK's voice. CSM-1B uses these for voice cloning.</p>
      </div>

      <div className="tab-body">
        <div className="grid-2">
          {/* Recording Panel */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <MicIcon /> Record Voice Sample
                </span>
              </div>

              <div className="form-group">
                <label className="form-label">What you'll say (optional)</label>
                <input
                  type="text"
                  className="form-input"
                  value={sampleText}
                  onChange={(e) => setSampleText(e.target.value)}
                  placeholder="e.g., Hello, my name is..."
                  disabled={isRecording}
                />
                <p className="form-hint">
                  Describe what you'll read aloud. This helps match voice to content.
                </p>
              </div>

              <div className="record-controls">
                <button
                  className={`record-btn-large ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  {isRecording ? <StopIcon /> : <MicLargeIcon />}
                </button>
                <div className="record-status">
                  <div className="record-status-label">
                    {isRecording ? `Recording... ${formatDuration(recordingDuration)}` : 'Ready to record'}
                  </div>
                  <div className="record-status-hint">
                    {isRecording
                      ? 'Click stop when done'
                      : 'Click to start recording your voice'}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <p className="form-hint">
                  <strong>Tips for PERFECT voice cloning:</strong>
                </p>
                <ul style={{ fontSize: '0.813rem', color: 'var(--text-muted)', paddingLeft: '1.25rem', marginTop: '0.5rem' }}>
                  <li><strong>LONGER is better!</strong> 30-60+ seconds per sample</li>
                  <li>Aim for 2-3 minutes TOTAL across all samples</li>
                  <li>Speak naturally at your normal pace</li>
                  <li>Record in a quiet environment</li>
                  <li>Say EXACTLY what you type in the text box</li>
                  <li>Varied content helps (questions, statements, emotions)</li>
                </ul>
              </div>
            </div>

            {/* Suggested Phrases */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <TextIcon /> Suggested Phrases
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {SUGGESTED_PHRASES.map((phrase, i) => (
                  <button
                    key={i}
                    className="btn btn-secondary"
                    onClick={() => setSampleText(phrase)}
                    style={{ justifyContent: 'flex-start', textAlign: 'left', fontSize: '0.813rem' }}
                    disabled={isRecording}
                  >
                    "{phrase}"
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Samples List */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <WaveformIcon /> Voice Samples
                  <span className="card-badge">{voiceSamples.length}</span>
                </span>
              </div>

              {voiceSamples.length === 0 ? (
                <div className="list-empty">
                  <div className="list-empty-icon"><MicOffIcon /></div>
                  <p>No voice samples yet.</p>
                  <p style={{ fontSize: '0.813rem', marginTop: '0.5rem' }}>
                    Record some samples to train NIVEK's voice.
                  </p>
                </div>
              ) : (
                <div>
                  {voiceSamples.map(sample => (
                    <div key={sample.id} className="list-item">
                      <div className="list-item-content">
                        <div className="list-item-title">{sample.text}</div>
                        <div className="list-item-meta">
                          {sample.duration}s &middot; {new Date(sample.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                      <div className="list-item-actions">
                        <button
                          className="btn-icon"
                          onClick={() => deleteVoiceSample(sample.id)}
                          title="Delete"
                        >
                          <TrashIcon />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Status */}
            <div className="card">
              <div className="card-header">
                <span className="card-title"><StatusIcon /> Voice Status</span>
              </div>
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Backend</span>
                  <span className="metric-value" style={{
                    color: connectionStatus === 'connected' ? 'var(--success)' : 'var(--text-muted)'
                  }}>
                    {connectionStatus}
                  </span>
                </div>
                <div className="metric">
                  <span className="metric-label">TTS Model</span>
                  <span className="metric-value">CSM-1B</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Sample Rate</span>
                  <span className="metric-value">24kHz</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Samples Loaded</span>
                  <span className="metric-value">{voiceSamples.length}</span>
                </div>
              </div>

              {voiceSamples.length > 0 && (
                <div className="status-success" style={{ marginTop: '1rem' }}>
                  <CheckIcon /> Voice cloning active with {voiceSamples.length} sample{voiceSamples.length !== 1 ? 's' : ''}
                </div>
              )}
            </div>

            {/* Info */}
            <div className="card">
              <div className="card-header">
                <span className="card-title"><InfoIcon /> How Voice Cloning Works</span>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <p style={{ marginBottom: '0.75rem' }}>
                  CSM-1B uses your voice samples as <strong>context</strong> to learn your unique voice
                  characteristics: tone, pitch, cadence, and speaking patterns.
                </p>
                <p style={{ marginBottom: '0.75rem' }}>
                  <strong>Key insight:</strong> The model needs enough audio to understand your voice.
                  Short clips (under 10s) often produce inconsistent results.
                </p>
                <p style={{ marginBottom: '0.75rem' }}>
                  <strong>Best results:</strong> Record 2-3 minutes of total audio across multiple samples.
                  The text you type should EXACTLY match what you say.
                </p>
                <p>
                  All samples are used together as context for generation. More = better!
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const SUGGESTED_PHRASES = [
  "Hello, my name is and I'm recording this voice sample for NIVEK. I want to make sure my voice sounds natural and authentic.",
  "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet, which helps capture my voice's full range.",
  "Sometimes I wonder about things. Like, what makes a good conversation? I think it's about really listening and responding thoughtfully.",
  "When I'm excited about something, my voice gets a bit faster and higher. But when I'm calm, I speak more slowly and deliberately.",
  "Let me think about that for a moment. Hmm, that's actually a really interesting point you've made there."
];

// === ICONS ===
function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}

function MicLargeIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2"/>
    </svg>
  );
}

function MicOffIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  );
}

function TextIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="4 7 4 4 20 4 20 7"/>
      <line x1="9" y1="20" x2="15" y2="20"/>
      <line x1="12" y1="4" x2="12" y2="20"/>
    </svg>
  );
}

function WaveformIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="4" y1="12" x2="4" y2="12"/>
      <line x1="8" y1="8" x2="8" y2="16"/>
      <line x1="12" y1="4" x2="12" y2="20"/>
      <line x1="16" y1="8" x2="16" y2="16"/>
      <line x1="20" y1="12" x2="20" y2="12"/>
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

function StatusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
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

function InfoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12.01" y2="8"/>
    </svg>
  );
}
