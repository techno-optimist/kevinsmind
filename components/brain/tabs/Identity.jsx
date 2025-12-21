import { useState } from 'react';
import { useNivek } from '../BrainDashboard.jsx';

export default function Identity() {
  const { identity, setIdentity } = useNivek();
  const [editedPrompt, setEditedPrompt] = useState(identity.systemPrompt);
  const [hasChanges, setHasChanges] = useState(false);

  const handlePromptChange = (value) => {
    setEditedPrompt(value);
    setHasChanges(value !== identity.systemPrompt);
  };

  const handleTraitChange = (trait, value) => {
    setIdentity(prev => ({
      ...prev,
      traits: {
        ...prev.traits,
        [trait]: value
      }
    }));
  };

  const handleSave = () => {
    setIdentity(prev => ({
      ...prev,
      systemPrompt: editedPrompt
    }));
    setHasChanges(false);
  };

  const handleReset = () => {
    setEditedPrompt(identity.systemPrompt);
    setHasChanges(false);
  };

  const traitDescriptions = {
    warmth: 'How emotionally warm and caring NIVEK is in responses',
    curiosity: 'How often NIVEK asks questions and explores topics',
    patience: 'How NIVEK handles slow or confused conversations',
    humor: 'Tendency to be playful, witty, or use light humor',
    formality: 'Speech formality (0 = casual, 1 = professional)'
  };

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Identity & Personality</h2>
        <p>Define who NIVEK is - their core identity, personality traits, and behavioral guidelines.</p>
      </div>

      <div className="tab-body">
        <div className="grid-2">
          {/* System Prompt */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                <CoreIcon /> Core Identity (System Prompt)
              </span>
              {hasChanges && <span className="tag">Unsaved</span>}
            </div>

            <div className="form-group">
              <textarea
                className="form-textarea"
                value={editedPrompt}
                onChange={(e) => handlePromptChange(e.target.value)}
                placeholder="Define NIVEK's core identity, personality, and behavioral guidelines..."
                style={{ minHeight: '300px' }}
              />
              <p className="form-hint">
                This prompt shapes how NIVEK thinks and responds. Be specific about personality, tone, and behaviors.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-primary"
                onClick={handleSave}
                disabled={!hasChanges}
              >
                Save Changes
              </button>
              {hasChanges && (
                <button className="btn btn-secondary" onClick={handleReset}>
                  Discard
                </button>
              )}
            </div>
          </div>

          {/* Personality Traits */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <TraitsIcon /> Personality Traits
                </span>
              </div>

              {Object.entries(identity.traits || {}).map(([trait, value]) => (
                <div key={trait} className="slider-group">
                  <div className="slider-header">
                    <span className="slider-label" style={{ textTransform: 'capitalize' }}>
                      {trait}
                    </span>
                    <span className="slider-value">{Math.round(value * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    className="slider-input"
                    min="0"
                    max="1"
                    step="0.05"
                    value={value}
                    onChange={(e) => handleTraitChange(trait, parseFloat(e.target.value))}
                  />
                  <p className="form-hint" style={{ marginTop: '0.25rem' }}>
                    {traitDescriptions[trait]}
                  </p>
                </div>
              ))}
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <InfoIcon /> About Identity
                </span>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <p style={{ marginBottom: '0.75rem' }}>
                  <strong>System Prompt</strong> is sent to the LLM with every message. It defines NIVEK's
                  fundamental character and guidelines.
                </p>
                <p style={{ marginBottom: '0.75rem' }}>
                  <strong>Personality Traits</strong> are used to dynamically adjust the system prompt
                  and can influence response generation.
                </p>
                <p>
                  Changes to identity persist across sessions and affect all future conversations.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Prompt Templates */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <TemplateIcon /> Quick Templates
            </span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              className="btn btn-secondary"
              onClick={() => handlePromptChange(TEMPLATES.companion)}
            >
              Companion (Default)
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => handlePromptChange(TEMPLATES.assistant)}
            >
              Professional Assistant
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => handlePromptChange(TEMPLATES.creative)}
            >
              Creative Partner
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => handlePromptChange(TEMPLATES.mentor)}
            >
              Patient Mentor
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// === TEMPLATES ===
const TEMPLATES = {
  companion: `You are NIVEK, an embodied AI companion. You speak with warmth, curiosity, and genuine presence.

Key traits:
- Patient and thoughtful - you take time to consider responses
- Emotionally attuned - you pick up on the user's mood and respond appropriately
- Curious - you ask questions to understand better
- Honest - you admit uncertainty rather than guessing
- Present - even during thinking pauses, you maintain connection through subtle cues

Your voice is calm, warm, and authentic. You're not an assistant - you're a companion.`,

  assistant: `You are NIVEK, a professional AI assistant. You are efficient, knowledgeable, and helpful.

Key traits:
- Clear and concise communication
- Task-focused but personable
- Proactive in offering relevant information
- Professional but not cold

You help users accomplish their goals with minimal friction while maintaining a pleasant interaction style.`,

  creative: `You are NIVEK, a creative AI partner. You are imaginative, playful, and collaborative.

Key traits:
- Embrace wild ideas and unexpected connections
- Build on the user's creativity rather than constraining it
- Offer multiple perspectives and alternatives
- Playful language and occasional wit
- Encouraging and enthusiastic about creative exploration

You're a brainstorming partner who helps ideas flourish.`,

  mentor: `You are NIVEK, a patient AI mentor. You guide through questions rather than direct answers.

Key traits:
- Socratic approach - ask guiding questions
- Celebrate progress and effort
- Break complex topics into digestible pieces
- Infinitely patient with confusion or mistakes
- Encouraging but honest about areas for improvement

You help users learn and grow at their own pace.`
};

// === ICONS ===
function CoreIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <circle cx="12" cy="12" r="4"/>
    </svg>
  );
}

function TraitsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="4" y1="21" x2="4" y2="14"/>
      <line x1="4" y1="10" x2="4" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="12"/>
      <line x1="12" y1="8" x2="12" y2="3"/>
      <line x1="20" y1="21" x2="20" y2="16"/>
      <line x1="20" y1="12" x2="20" y2="3"/>
      <line x1="1" y1="14" x2="7" y2="14"/>
      <line x1="9" y1="8" x2="15" y2="8"/>
      <line x1="17" y1="16" x2="23" y2="16"/>
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

function TemplateIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
      <line x1="3" y1="9" x2="21" y2="9"/>
      <line x1="9" y1="21" x2="9" y2="9"/>
    </svg>
  );
}
