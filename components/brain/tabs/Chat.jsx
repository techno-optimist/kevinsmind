import { useState, useRef, useEffect, useCallback } from 'react';
import { useNivek } from '../BrainDashboard.jsx';
import '../Chat.css';

const FILLER_TYPES = ['breath', 'pulse', 'text', 'sway', 'none'];

// Robot expression mapping based on response sentiment/content
const EXPRESSION_KEYWORDS = {
  happy: ['happy', 'glad', 'great', 'wonderful', 'excited', 'love', 'enjoy', '!'],
  curious: ['interesting', 'wonder', 'curious', 'tell me', 'how', 'why', '?'],
  thinking: ['hmm', 'let me think', 'considering', 'perhaps', 'maybe'],
  surprise: ['wow', 'amazing', 'incredible', 'unexpected', 'surprising'],
  sad: ['sorry', 'unfortunately', 'sad', 'difficult', 'hard'],
  nod: ['yes', 'agree', 'right', 'exactly', 'correct', 'indeed'],
  shake: ['no', 'don\'t', 'won\'t', 'can\'t', 'disagree'],
};

function detectExpressionFromText(text) {
  const lowerText = text.toLowerCase();
  for (const [expression, keywords] of Object.entries(EXPRESSION_KEYWORDS)) {
    for (const keyword of keywords) {
      if (lowerText.includes(keyword)) {
        return expression;
      }
    }
  }
  return 'neutral';
}

export default function Chat() {
  const {
    wsRef,
    connectionStatus,
    identity,
    voiceSamples,
    settings,
    memories,
    addMemory,
    currentSession,
    addMessageToSession,
    clearCurrentSession,
    conversations,
    loadConversation
  } = useNivek();

  // Use currentSession.messages as the source of truth
  const messages = currentSession.messages;

  const [isThinking, setIsThinking] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [fillerType, setFillerType] = useState(settings?.fillerType || 'breath');
  const [transcript, setTranscript] = useState('');
  const [thinkingStartTime, setThinkingStartTime] = useState(null);
  const [robotWsRef, setRobotWsRef] = useState(null);
  const [metrics, setMetrics] = useState({
    lastLatency: 0,
    avgLatency: 0,
    count: 0,
    lastAudioDuration: 0
  });

  const messagesEndRef = useRef(null);
  const audioRef = useRef(null);
  const recognitionRef = useRef(null);
  const handleSendRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  // Initialize speech recognition
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        setTranscript(finalTranscript || interimTranscript);

        if (finalTranscript) {
          // Use ref to avoid stale closure
          if (handleSendRef.current) {
            handleSendRef.current(finalTranscript);
          }
          setTranscript('');
          setIsListening(false);
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setTranscript('');
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  // Robot WebSocket connection (for expression sync)
  useEffect(() => {
    const robotSettings = settings?.robot || {};
    if (!robotSettings.syncWithChat) return;

    const host = robotSettings.host || 'localhost';
    const port = robotSettings.port || 8001;

    let ws = null;
    let reconnectTimeout = null;

    const connect = () => {
      ws = new WebSocket(`ws://${host}:${port}/robot`);

      ws.onopen = () => {
        console.log('[Robot] Connected for expression sync');
        setRobotWsRef(ws);
      };

      ws.onclose = () => {
        setRobotWsRef(null);
        // Try to reconnect in 5s
        reconnectTimeout = setTimeout(connect, 5000);
      };

      ws.onerror = () => {
        // Silent fail - robot connection is optional
      };
    };

    connect();

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, [settings?.robot?.syncWithChat, settings?.robot?.host, settings?.robot?.port]);

  // Function to trigger robot expression
  const triggerRobotExpression = useCallback((expression) => {
    if (robotWsRef?.readyState === WebSocket.OPEN) {
      robotWsRef.send(JSON.stringify({
        type: 'play_expression',
        expression: expression
      }));
    }
  }, [robotWsRef]);

  // WebSocket message handler
  useEffect(() => {
    if (!wsRef) return;

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'thinking':
          // Trigger thinking expression on robot
          if (settings?.robot?.syncWithChat) {
            triggerRobotExpression('thinking');
          }
          break;

        case 'response_start':
          break;

        case 'audio_chunk':
          playAudio(data.data, data.sample_rate);
          break;

        case 'response_end':
          const elapsed = Date.now() - (thinkingStartTime || Date.now());
          setMetrics(prev => ({
            lastLatency: data.latency_ms || elapsed,
            avgLatency: Math.round((prev.avgLatency * prev.count + (data.latency_ms || elapsed)) / (prev.count + 1)),
            count: prev.count + 1,
            lastAudioDuration: prev.lastAudioDuration
          }));

          setIsThinking(false);
          setThinkingStartTime(null);

          // Detect and trigger appropriate robot expression
          if (settings?.robot?.syncWithChat && data.text) {
            const expression = detectExpressionFromText(data.text);
            triggerRobotExpression(expression);
          }

          addMessageToSession({
            role: 'assistant',
            content: data.text,
            timestamp: Date.now()
          });
          break;

        case 'error':
          console.error('Server error:', data.message);
          setIsThinking(false);
          setThinkingStartTime(null);
          break;
      }
    };

    wsRef.addEventListener('message', handleMessage);
    return () => wsRef.removeEventListener('message', handleMessage);
  }, [wsRef, thinkingStartTime, addMessageToSession, settings?.robot?.syncWithChat, triggerRobotExpression]);

  const playAudio = async (base64Data, sampleRate) => {
    try {
      setIsSpeaking(true);

      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);

      setMetrics(prev => ({
        ...prev,
        lastAudioDuration: Math.round(audioBuffer.duration * 1000)
      }));

      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.onended = () => {
        setIsSpeaking(false);
      };
      source.start(0);

      audioRef.current = { context: audioContext, source };

    } catch (err) {
      console.error('Audio playback error:', err);
      setIsSpeaking(false);
    }
  };

  const handleSend = useCallback(async (text) => {
    if (!text.trim() || isThinking) return;

    addMessageToSession({
      role: 'user',
      content: text,
      timestamp: Date.now()
    });

    setIsThinking(true);
    const startTime = Date.now();
    setThinkingStartTime(startTime);

    if (wsRef?.readyState === WebSocket.OPEN) {
      // Build context from memories and identity
      const context = {
        systemPrompt: identity?.systemPrompt || '',
        memories: memories?.map(m => `${m.type}: ${m.content}`).join('\n') || '',
        traits: identity?.traits || {}
      };

      // LLM settings
      const llmSettings = {
        provider: settings?.llmProvider || 'mock',
        model: settings?.llmModel,
        apiKey: settings?.apiKey
      };

      wsRef.send(JSON.stringify({
        type: 'message',
        text: text,
        mock_audio: !settings?.ttsEnabled,
        context: context,
        llm: llmSettings,
        voice_samples: voiceSamples.map(s => ({ text: s.text, audio: s.audio, format: s.format }))
      }));
    } else {
      // Fallback: mock response
      await new Promise(r => setTimeout(r, 1500));

      const response = generateMockResponse(text);
      const elapsed = Date.now() - startTime;

      setMetrics(prev => ({
        lastLatency: elapsed,
        avgLatency: Math.round((prev.avgLatency * prev.count + elapsed) / (prev.count + 1)),
        count: prev.count + 1,
        lastAudioDuration: 0
      }));

      setIsThinking(false);
      setThinkingStartTime(null);

      addMessageToSession({
        role: 'assistant',
        content: response,
        timestamp: Date.now()
      });
    }
  }, [isThinking, wsRef, identity, memories, voiceSamples, settings, addMessageToSession]);

  // Keep handleSendRef updated for speech recognition callback
  useEffect(() => {
    handleSendRef.current = handleSend;
  }, [handleSend]);

  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      alert('Speech recognition is not supported in this browser. Please use Chrome.');
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      setTranscript('');
    } else {
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const clearConversation = () => {
    clearCurrentSession(true); // true = save first
  };

  // Save a message to memory
  const saveToMemory = useCallback((message, memoryType = 'conversation') => {
    addMemory({
      type: memoryType,
      content: message.content,
      source: message.role === 'user' ? 'user_said' : 'nivek_said',
      context: message.role === 'user'
        ? 'Something the user shared'
        : 'Something NIVEK said that may be worth remembering'
    });
  }, [addMemory]);

  return (
    <div className="chat-panel">
      <div className="chat-main">
        <div className="chat-container">
          <MessageList messages={messages} onSaveToMemory={saveToMemory} />

          {isListening && (
            <ListeningIndicator transcript={transcript} />
          )}

          {isThinking && (
            <ThinkingIndicator
              type={fillerType}
              startTime={thinkingStartTime}
            />
          )}

          {isSpeaking && (
            <SpeakingIndicator />
          )}

          <div ref={messagesEndRef} />

          <Input
            onSend={handleSend}
            disabled={isThinking || isSpeaking || isListening}
            onMicClick={toggleListening}
            isListening={isListening}
            transcript={transcript}
          />
        </div>
      </div>

      <aside className="chat-sidebar">
        {/* Session Controls */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Session</span>
            <span className="tag tag-primary">{messages.length} msgs</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button
              className="btn btn-primary"
              onClick={() => clearCurrentSession(false)}
              style={{ flex: 1, fontSize: '0.75rem' }}
            >
              <NewChatIcon /> New Chat
            </button>
            {messages.length > 0 && (
              <button
                className="btn btn-secondary"
                onClick={clearConversation}
                style={{ flex: 1, fontSize: '0.75rem' }}
              >
                Save & Clear
              </button>
            )}
          </div>
        </div>

        {/* Recent Conversations */}
        {conversations.length > 0 && (
          <div className="card">
            <div className="card-header">
              <span className="card-title"><HistoryIcon /> History</span>
            </div>
            <div className="conversation-list">
              {conversations.slice(0, 5).map(conv => (
                <button
                  key={conv.id}
                  className={`conversation-item ${currentSession.loadedFrom === conv.id ? 'active' : ''}`}
                  onClick={() => loadConversation(conv.id)}
                >
                  <span className="conversation-preview">{conv.preview}</span>
                  <span className="conversation-meta">{conv.messageCount} msgs</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="card">
          <div className="card-header">
            <span className="card-title">Connection</span>
            <span className={`tag ${connectionStatus === 'connected' ? 'tag-primary' : ''}`}>
              {connectionStatus}
            </span>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Filler Animation</span>
          </div>
          <div className="filler-options">
            {FILLER_TYPES.map(type => (
              <button
                key={type}
                className={`filler-btn ${fillerType === type ? 'active' : ''}`}
                onClick={() => setFillerType(type)}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Metrics</span>
          </div>
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Last latency</span>
              <span className="metric-value">{metrics.lastLatency}ms</span>
            </div>
            <div className="metric">
              <span className="metric-label">Avg latency</span>
              <span className="metric-value">{metrics.avgLatency}ms</span>
            </div>
            <div className="metric">
              <span className="metric-label">Audio duration</span>
              <span className="metric-value">{metrics.lastAudioDuration}ms</span>
            </div>
            <div className="metric">
              <span className="metric-label">Messages</span>
              <span className="metric-value">{metrics.count}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Context</span>
          </div>
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Memories</span>
              <span className="metric-value">{memories?.length || 0}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Voice Samples</span>
              <span className="metric-value">{voiceSamples?.length || 0}</span>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}

// === SUB-COMPONENTS ===

function MessageList({ messages, onSaveToMemory }) {
  const [savedMessages, setSavedMessages] = useState(new Set());

  const handleSaveToMemory = (msg, index) => {
    onSaveToMemory(msg);
    setSavedMessages(prev => new Set([...prev, index]));
  };

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <p>Start a conversation with NIVEK.</p>
        <p className="hint">Click the microphone or type a message.</p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((msg, i) => (
        <div key={msg.id || i} className={`message ${msg.role}`}>
          <div className="message-content">{msg.content}</div>
          <div className="message-actions">
            <button
              className={`save-memory-btn ${savedMessages.has(i) ? 'saved' : ''}`}
              onClick={() => handleSaveToMemory(msg, i)}
              disabled={savedMessages.has(i)}
              title={savedMessages.has(i) ? 'Saved to memory' : 'Save to memory'}
            >
              {savedMessages.has(i) ? <CheckIcon /> : <MemoryIcon />}
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function ListeningIndicator({ transcript }) {
  return (
    <div className="listening-container">
      <div className="listening-pulse" />
      <div className="listening-content">
        <span className="listening-label">Listening...</span>
        {transcript && <span className="transcript">{transcript}</span>}
      </div>
    </div>
  );
}

function ThinkingIndicator({ type, startTime }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      if (startTime) {
        setElapsed(Date.now() - startTime);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [startTime]);

  return (
    <div className="thinking-container">
      <div className={`thinking-indicator thinking-${type}`}>
        {renderThinkingContent(type)}
      </div>
      <span className="elapsed-time">{(elapsed / 1000).toFixed(1)}s</span>
    </div>
  );
}

function renderThinkingContent(type) {
  switch (type) {
    case 'breath':
      return <div className="breath-container"><div className="breath-circle" /></div>;
    case 'pulse':
      return (
        <div className="pulse-container">
          <div className="pulse-ring pulse-ring-1" />
          <div className="pulse-ring pulse-ring-2" />
          <div className="pulse-ring pulse-ring-3" />
          <div className="pulse-core" />
        </div>
      );
    case 'text':
      return <TextFiller />;
    case 'sway':
      return <div className="sway-container"><div className="sway-dot" /></div>;
    case 'none':
      return null;
    default:
      return null;
  }
}

function TextFiller() {
  const [dots, setDots] = useState(0);
  const [filler] = useState(() => {
    const fillers = ['Hmm', 'Let me think', 'Considering'];
    return fillers[Math.floor(Math.random() * fillers.length)];
  });

  useEffect(() => {
    const interval = setInterval(() => setDots(d => (d + 1) % 4), 400);
    return () => clearInterval(interval);
  }, []);

  return <div className="text-filler">{filler}{'.'.repeat(dots)}</div>;
}

function SpeakingIndicator() {
  return (
    <div className="speaking-container">
      <div className="speaking-bars">
        <div className="bar bar-1" />
        <div className="bar bar-2" />
        <div className="bar bar-3" />
        <div className="bar bar-4" />
        <div className="bar bar-5" />
      </div>
      <span className="speaking-label">Speaking...</span>
    </div>
  );
}

function Input({ onSend, disabled, onMicClick, isListening, transcript }) {
  const [text, setText] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (!disabled && !isListening) {
      inputRef.current?.focus();
    }
  }, [disabled, isListening]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onSend(text);
      setText('');
    }
  };

  return (
    <form className="input-container" onSubmit={handleSubmit}>
      <button
        type="button"
        onClick={onMicClick}
        disabled={disabled && !isListening}
        className={`mic-btn ${isListening ? 'listening' : ''}`}
        title={isListening ? 'Stop listening' : 'Start voice input'}
      >
        <MicIcon />
      </button>
      <input
        ref={inputRef}
        type="text"
        value={isListening ? transcript : text}
        onChange={(e) => setText(e.target.value)}
        placeholder={isListening ? 'Listening...' : disabled ? 'Waiting...' : 'Type or click mic to speak...'}
        disabled={disabled || isListening}
        className="message-input"
      />
      <button type="submit" disabled={disabled || !text.trim() || isListening} className="send-btn">
        Send
      </button>
    </form>
  );
}

function MicIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function MemoryIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
      <line x1="8" y1="21" x2="16" y2="21"/>
      <line x1="12" y1="17" x2="12" y2="21"/>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function NewChatIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="5" x2="12" y2="19"/>
      <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  );
}

function generateMockResponse(input) {
  const responses = [
    `I've been thinking about what you said: "${input.slice(0, 30)}${input.length > 30 ? '...' : ''}"`,
    `That's an interesting point. ${input.includes('?') ? 'Let me consider that question.' : 'I appreciate you sharing that.'}`,
    `Hmm, when you mention that, it reminds me of our earlier conversation patterns.`,
    `I'm processing your words. There's something meaningful in what you're exploring.`,
    `That resonates with me in an interesting way. Tell me more about what you mean.`,
  ];
  return responses[Math.floor(Math.random() * responses.length)];
}
