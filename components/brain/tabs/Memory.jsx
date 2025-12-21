import { useState } from 'react';
import { useNivek } from '../BrainDashboard.jsx';

const MEMORY_TYPES = [
  { value: 'fact', label: 'Fact', description: 'Things NIVEK knows about you' },
  { value: 'preference', label: 'Preference', description: 'Your likes, dislikes, preferences' },
  { value: 'context', label: 'Context', description: 'Background information for conversations' },
  { value: 'instruction', label: 'Instruction', description: 'Specific ways NIVEK should behave' }
];

export default function Memory() {
  const { memories, addMemory, updateMemory, deleteMemory, conversations } = useNivek();
  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [newMemory, setNewMemory] = useState({ type: 'fact', content: '' });
  const [filter, setFilter] = useState('all');

  const handleAdd = () => {
    if (!newMemory.content.trim()) return;
    addMemory(newMemory);
    setNewMemory({ type: 'fact', content: '' });
    setIsAdding(false);
  };

  const handleUpdate = (id) => {
    const memory = memories.find(m => m.id === id);
    if (memory) {
      updateMemory(id, { content: memory.content });
    }
    setEditingId(null);
  };

  const filteredMemories = filter === 'all'
    ? memories
    : memories.filter(m => m.type === filter);

  const memoryStats = {
    total: memories.length,
    facts: memories.filter(m => m.type === 'fact').length,
    preferences: memories.filter(m => m.type === 'preference').length,
    context: memories.filter(m => m.type === 'context').length,
    instructions: memories.filter(m => m.type === 'instruction').length
  };

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Memory Bank</h2>
        <p>Long-term memories that persist across conversations. NIVEK uses these to personalize interactions.</p>
      </div>

      <div className="tab-body">
        <div className="grid-2">
          {/* Memory List */}
          <div>
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <MemoryIcon /> Memories
                  <span className="card-badge">{memories.length}</span>
                </span>
                <button className="btn btn-primary" onClick={() => setIsAdding(true)}>
                  + Add
                </button>
              </div>

              {/* Filter */}
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                <button
                  className={`btn ${filter === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setFilter('all')}
                  style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}
                >
                  All ({memoryStats.total})
                </button>
                {MEMORY_TYPES.map(type => (
                  <button
                    key={type.value}
                    className={`btn ${filter === type.value ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setFilter(type.value)}
                    style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}
                  >
                    {type.label} ({memoryStats[type.value + 's'] || memoryStats[type.value] || 0})
                  </button>
                ))}
              </div>

              {/* Add Form */}
              {isAdding && (
                <div style={{
                  padding: '1rem',
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--accent)',
                  borderRadius: '8px',
                  marginBottom: '1rem'
                }}>
                  <div className="form-group">
                    <label className="form-label">Type</label>
                    <select
                      className="form-select"
                      value={newMemory.type}
                      onChange={(e) => setNewMemory(prev => ({ ...prev, type: e.target.value }))}
                    >
                      {MEMORY_TYPES.map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Content</label>
                    <textarea
                      className="form-textarea"
                      value={newMemory.content}
                      onChange={(e) => setNewMemory(prev => ({ ...prev, content: e.target.value }))}
                      placeholder="Enter the memory content..."
                      style={{ minHeight: '80px' }}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="btn btn-primary" onClick={handleAdd}>Save</button>
                    <button className="btn btn-secondary" onClick={() => setIsAdding(false)}>Cancel</button>
                  </div>
                </div>
              )}

              {/* Memory List */}
              {filteredMemories.length === 0 ? (
                <div className="list-empty">
                  <div className="list-empty-icon"><BrainEmptyIcon /></div>
                  <p>No memories yet.</p>
                  <p style={{ fontSize: '0.813rem', marginTop: '0.5rem' }}>
                    Add facts, preferences, and context to help NIVEK remember you.
                  </p>
                </div>
              ) : (
                <div>
                  {filteredMemories.map(memory => (
                    <div key={memory.id} className="list-item">
                      <div className="list-item-content">
                        {editingId === memory.id ? (
                          <textarea
                            className="form-textarea"
                            value={memory.content}
                            onChange={(e) => updateMemory(memory.id, { content: e.target.value })}
                            style={{ minHeight: '60px', marginBottom: '0.5rem' }}
                            autoFocus
                          />
                        ) : (
                          <div className="list-item-title">{memory.content}</div>
                        )}
                        <div className="list-item-meta">
                          <span className="tag">{memory.type}</span>
                          {' '}&middot;{' '}
                          {new Date(memory.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                      <div className="list-item-actions">
                        {editingId === memory.id ? (
                          <>
                            <button className="btn-icon" onClick={() => handleUpdate(memory.id)} title="Save">
                              <CheckIcon />
                            </button>
                            <button className="btn-icon" onClick={() => setEditingId(null)} title="Cancel">
                              <XIcon />
                            </button>
                          </>
                        ) : (
                          <>
                            <button className="btn-icon" onClick={() => setEditingId(memory.id)} title="Edit">
                              <EditIcon />
                            </button>
                            <button className="btn-icon" onClick={() => deleteMemory(memory.id)} title="Delete">
                              <TrashIcon />
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right Column */}
          <div>
            {/* Quick Add */}
            <div className="card">
              <div className="card-header">
                <span className="card-title"><BoltIcon /> Quick Add</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <QuickAddButton
                  label="My name is..."
                  onClick={() => {
                    setNewMemory({ type: 'fact', content: 'User\'s name is ' });
                    setIsAdding(true);
                  }}
                />
                <QuickAddButton
                  label="I prefer..."
                  onClick={() => {
                    setNewMemory({ type: 'preference', content: 'User prefers ' });
                    setIsAdding(true);
                  }}
                />
                <QuickAddButton
                  label="When I say X, I mean..."
                  onClick={() => {
                    setNewMemory({ type: 'context', content: 'When user says "...", they mean ' });
                    setIsAdding(true);
                  }}
                />
                <QuickAddButton
                  label="Always/Never..."
                  onClick={() => {
                    setNewMemory({ type: 'instruction', content: 'Always ' });
                    setIsAdding(true);
                  }}
                />
              </div>
            </div>

            {/* Conversation History */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">
                  <HistoryIcon /> Recent Conversations
                  <span className="card-badge">{conversations?.length || 0}</span>
                </span>
              </div>
              {!conversations || conversations.length === 0 ? (
                <p style={{ fontSize: '0.813rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>
                  No saved conversations yet.
                </p>
              ) : (
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {conversations.slice(0, 10).map(conv => (
                    <div key={conv.id} className="list-item">
                      <div className="list-item-content">
                        <div className="list-item-title" style={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {conv.preview}
                        </div>
                        <div className="list-item-meta">
                          {conv.messageCount} messages &middot; {new Date(conv.createdAt).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Stats */}
            <div className="card">
              <div className="card-header">
                <span className="card-title"><StatsIcon /> Memory Stats</span>
              </div>
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Total Memories</span>
                  <span className="metric-value">{memoryStats.total}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Facts</span>
                  <span className="metric-value">{memoryStats.facts}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Preferences</span>
                  <span className="metric-value">{memoryStats.preferences}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Context Items</span>
                  <span className="metric-value">{memoryStats.context}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Instructions</span>
                  <span className="metric-value">{memoryStats.instructions}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function QuickAddButton({ label, onClick }) {
  return (
    <button
      className="btn btn-secondary"
      onClick={onClick}
      style={{ justifyContent: 'flex-start', textAlign: 'left' }}
    >
      + {label}
    </button>
  );
}

// === ICONS ===
function MemoryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
      <line x1="8" y1="21" x2="16" y2="21"/>
      <line x1="12" y1="17" x2="12" y2="21"/>
    </svg>
  );
}

function BrainEmptyIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 2a8 8 0 0 0-8 8c0 2.5 1.5 4.5 3 6s2 3 2 4h6c0-1 .5-2.5 2-4s3-3.5 3-6a8 8 0 0 0-8-8z"/>
      <circle cx="12" cy="10" r="2"/>
    </svg>
  );
}

function EditIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
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

function BoltIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
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

function StatsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>
  );
}
