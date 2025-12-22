import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import Chat from './tabs/Chat.jsx';
import Identity from './tabs/Identity.jsx';
import Memory from './tabs/Memory.jsx';
import Voice from './tabs/Voice.jsx';
import Robot from './tabs/Robot.jsx';
import Settings from './tabs/Settings.jsx';
import './BrainDashboard.css';

// === NIVEK CONTEXT ===
// Shared state across all tabs
const NivekContext = createContext(null);

export function useNivek() {
  return useContext(NivekContext);
}

// Default personality/identity
const DEFAULT_IDENTITY = {
  name: 'NIVEK',
  systemPrompt: `You are NIVEK, an embodied AI companion. You speak with warmth, curiosity, and genuine presence.

Key traits:
- Patient and thoughtful - you take time to consider responses
- Emotionally attuned - you pick up on the user's mood and respond appropriately
- Curious - you ask questions to understand better
- Honest - you admit uncertainty rather than guessing
- Present - even during thinking pauses, you maintain connection through subtle cues

Your voice is calm, warm, and authentic. You're not an assistant - you're a companion.`,
  traits: {
    warmth: 0.8,
    curiosity: 0.7,
    patience: 0.9,
    humor: 0.4,
    formality: 0.3
  }
};

// Default settings
const DEFAULT_SETTINGS = {
  llmProvider: 'claude',
  llmModel: 'claude-sonnet-4-20250514',
  apiKey: '',
  ttsEnabled: true,
  ttsModel: 'csm-1b',
  fillerType: 'breath',
  autoSave: true
};

export default function BrainDashboard() {
  // Connection state
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [wsRef, setWsRef] = useState(null);

  // Core NIVEK state
  const [identity, setIdentity] = useState(() => {
    const saved = localStorage.getItem('nivek_identity');
    return saved ? JSON.parse(saved) : DEFAULT_IDENTITY;
  });

  const [memories, setMemories] = useState(() => {
    const saved = localStorage.getItem('nivek_memories');
    return saved ? JSON.parse(saved) : [];
  });

  const [voiceSamples, setVoiceSamples] = useState(() => {
    const saved = localStorage.getItem('nivek_voice_samples');
    return saved ? JSON.parse(saved) : [];
  });

  const [conversations, setConversations] = useState(() => {
    const saved = localStorage.getItem('nivek_conversations');
    return saved ? JSON.parse(saved) : [];
  });

  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('nivek_settings');
    return saved ? JSON.parse(saved) : DEFAULT_SETTINGS;
  });

  // Current chat session - persisted across tab switches and page reloads
  const [currentSession, setCurrentSession] = useState(() => {
    const saved = localStorage.getItem('nivek_current_session');
    return saved ? JSON.parse(saved) : { id: Date.now(), messages: [], startedAt: new Date().toISOString() };
  });

  // Persist state to localStorage
  useEffect(() => {
    localStorage.setItem('nivek_identity', JSON.stringify(identity));
  }, [identity]);

  useEffect(() => {
    localStorage.setItem('nivek_memories', JSON.stringify(memories));
  }, [memories]);

  useEffect(() => {
    localStorage.setItem('nivek_voice_samples', JSON.stringify(voiceSamples));
  }, [voiceSamples]);

  useEffect(() => {
    localStorage.setItem('nivek_conversations', JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    localStorage.setItem('nivek_settings', JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    localStorage.setItem('nivek_current_session', JSON.stringify(currentSession));
  }, [currentSession]);

  // WebSocket connection
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let isCleaningUp = false;

    const connect = () => {
      if (isCleaningUp) return;

      setConnectionStatus('connecting');
      ws = new WebSocket('ws://localhost:8000/ws');

      ws.onopen = () => {
        if (isCleaningUp) {
          ws.close();
          return;
        }
        setConnectionStatus('connected');
        setWsRef(ws);
        console.log('WebSocket connected');
      };

      ws.onclose = () => {
        if (isCleaningUp) return;
        setConnectionStatus('disconnected');
        setWsRef(null);
        reconnectTimeout = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        if (isCleaningUp) return;
        setConnectionStatus('error');
      };
    };

    connect();

    return () => {
      isCleaningUp = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  // Memory management
  const addMemory = useCallback((memory) => {
    const newMemory = {
      id: Date.now(),
      createdAt: new Date().toISOString(),
      ...memory
    };
    setMemories(prev => [...prev, newMemory]);
  }, []);

  const updateMemory = useCallback((id, updates) => {
    setMemories(prev => prev.map(m =>
      m.id === id ? { ...m, ...updates, updatedAt: new Date().toISOString() } : m
    ));
  }, []);

  const deleteMemory = useCallback((id) => {
    setMemories(prev => prev.filter(m => m.id !== id));
  }, []);

  // Voice sample management
  const addVoiceSample = useCallback((sample) => {
    const newSample = {
      id: Date.now(),
      createdAt: new Date().toISOString(),
      ...sample
    };
    setVoiceSamples(prev => [...prev, newSample]);
  }, []);

  const deleteVoiceSample = useCallback((id) => {
    setVoiceSamples(prev => prev.filter(s => s.id !== id));
  }, []);

  // Conversation management
  const saveConversation = useCallback((messages) => {
    if (messages.length === 0) return;

    const conversation = {
      id: Date.now(),
      createdAt: new Date().toISOString(),
      preview: messages[0]?.content?.slice(0, 50) || 'New conversation',
      messageCount: messages.length,
      messages
    };
    setConversations(prev => [conversation, ...prev].slice(0, 50)); // Keep last 50
  }, []);

  // Current session management
  const addMessageToSession = useCallback((message) => {
    setCurrentSession(prev => ({
      ...prev,
      messages: [...prev.messages, { ...message, id: Date.now() }]
    }));
  }, []);

  const clearCurrentSession = useCallback((saveFirst = true) => {
    if (saveFirst && currentSession.messages.length > 0) {
      saveConversation(currentSession.messages);
    }
    setCurrentSession({
      id: Date.now(),
      messages: [],
      startedAt: new Date().toISOString()
    });
  }, [currentSession.messages, saveConversation]);

  const loadConversation = useCallback((conversationId) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      // Save current session first if it has messages
      if (currentSession.messages.length > 0) {
        saveConversation(currentSession.messages);
      }
      setCurrentSession({
        id: conversation.id,
        messages: conversation.messages,
        startedAt: conversation.createdAt,
        loadedFrom: conversationId
      });
    }
  }, [conversations, currentSession.messages, saveConversation]);

  // Context value
  const contextValue = {
    // Connection
    connectionStatus,
    wsRef,

    // Identity
    identity,
    setIdentity,

    // Memories
    memories,
    addMemory,
    updateMemory,
    deleteMemory,

    // Voice
    voiceSamples,
    addVoiceSample,
    deleteVoiceSample,

    // Conversations
    conversations,
    saveConversation,

    // Current session
    currentSession,
    addMessageToSession,
    clearCurrentSession,
    loadConversation,

    // Settings
    settings,
    setSettings
  };

  return (
    <NivekContext.Provider value={contextValue}>
      <div className="brain-dashboard">
        <header className="brain-header">
          <div className="brain-title">
            <div className="brain-icon">
              <BrainIcon />
            </div>
            <div>
              <h1>NIVEK</h1>
              <span className="brain-subtitle">Brain Dashboard</span>
            </div>
          </div>

          <ConnectionIndicator status={connectionStatus} />
        </header>

        <nav className="brain-nav">
          <NavLink to="/brain/chat" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <ChatIcon /> Chat
          </NavLink>
          <NavLink to="/brain/identity" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <IdentityIcon /> Identity
          </NavLink>
          <NavLink to="/brain/memory" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <MemoryIcon /> Memory
          </NavLink>
          <NavLink to="/brain/voice" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <VoiceIcon /> Voice
          </NavLink>
          <NavLink to="/brain/robot" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <RobotIcon /> Robot
          </NavLink>
          <NavLink to="/brain/settings" className={({ isActive }) => isActive ? 'nav-tab active' : 'nav-tab'}>
            <SettingsIcon /> Settings
          </NavLink>
        </nav>

        <main className="brain-content">
          <Routes>
            <Route path="chat" element={<Chat />} />
            <Route path="identity" element={<Identity />} />
            <Route path="memory" element={<Memory />} />
            <Route path="voice" element={<Voice />} />
            <Route path="robot" element={<Robot />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/brain/chat" replace />} />
          </Routes>
        </main>
      </div>
    </NivekContext.Provider>
  );
}

// === CONNECTION INDICATOR ===
function ConnectionIndicator({ status }) {
  const colors = {
    connected: '#22c55e',
    connecting: '#eab308',
    disconnected: '#ef4444',
    error: '#ef4444'
  };

  return (
    <div className="connection-indicator">
      <span className="connection-dot" style={{ backgroundColor: colors[status] }} />
      <span className="connection-text">{status}</span>
    </div>
  );
}

// === ICONS ===
function BrainIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 2a8 8 0 0 0-8 8c0 2.5 1.5 4.5 3 6s2 3 2 4h6c0-1 .5-2.5 2-4s3-3.5 3-6a8 8 0 0 0-8-8z"/>
      <path d="M12 2v4M8 6l2 2M16 6l-2 2M9 20h6M10 22h4"/>
      <circle cx="12" cy="10" r="2"/>
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
    </svg>
  );
}

function IdentityIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="8" r="4"/>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
    </svg>
  );
}

function MemoryIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
      <line x1="8" y1="21" x2="16" y2="21"/>
      <line x1="12" y1="17" x2="12" y2="21"/>
    </svg>
  );
}

function VoiceIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  );
}

function RobotIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="11" width="18" height="10" rx="2"/>
      <circle cx="12" cy="5" r="3"/>
      <line x1="12" y1="8" x2="12" y2="11"/>
      <circle cx="8" cy="16" r="1"/>
      <circle cx="16" cy="16" r="1"/>
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
    </svg>
  );
}
