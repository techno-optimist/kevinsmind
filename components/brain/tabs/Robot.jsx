import { useState, useEffect, useCallback, useRef } from 'react';
import { useNivek } from '../BrainDashboard.jsx';

// Robot connection states
const CONNECTION_STATES = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error'
};

export default function Robot() {
  const { settings, setSettings } = useNivek();

  // Robot connection state
  const [robotStatus, setRobotStatus] = useState(CONNECTION_STATES.DISCONNECTED);
  const [robotInfo, setRobotInfo] = useState(null);
  const [lastError, setLastError] = useState(null);
  const wsRef = useRef(null);

  // Motor state
  const [motorState, setMotorState] = useState({
    enabled: false,
    gravityCompensation: false
  });

  // Head pose controls
  const [headPose, setHeadPose] = useState({
    x: 0,      // mm forward/back
    y: 0,      // mm left/right
    z: 0,      // mm up/down
    roll: 0,   // degrees
    pitch: 0,  // degrees
    yaw: 0     // degrees
  });

  // Antenna positions
  const [antennas, setAntennas] = useState({
    left: 0,   // radians
    right: 0   // radians
  });

  // Body yaw
  const [bodyYaw, setBodyYaw] = useState(0);

  // Expression presets
  const [currentExpression, setCurrentExpression] = useState('neutral');

  // Robot settings from global settings
  const robotSettings = settings.robot || {
    host: 'localhost',
    port: 8001,
    autoConnect: false,
    syncWithChat: true
  };

  const updateRobotSettings = (key, value) => {
    setSettings(prev => ({
      ...prev,
      robot: {
        ...prev.robot,
        [key]: value
      }
    }));
  };

  // WebSocket connection to robot bridge
  const connectToRobot = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    setRobotStatus(CONNECTION_STATES.CONNECTING);
    setLastError(null);

    const ws = new WebSocket(`ws://${robotSettings.host}:${robotSettings.port}/robot`);

    ws.onopen = () => {
      setRobotStatus(CONNECTION_STATES.CONNECTED);
      wsRef.current = ws;
      // Request robot info
      ws.send(JSON.stringify({ type: 'get_info' }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleRobotMessage(data);
      } catch (err) {
        console.error('Failed to parse robot message:', err);
      }
    };

    ws.onclose = () => {
      setRobotStatus(CONNECTION_STATES.DISCONNECTED);
      wsRef.current = null;
    };

    ws.onerror = (error) => {
      setRobotStatus(CONNECTION_STATES.ERROR);
      setLastError('Connection failed. Is the robot bridge running?');
    };
  }, [robotSettings.host, robotSettings.port]);

  const disconnectFromRobot = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const handleRobotMessage = (data) => {
    switch (data.type) {
      case 'info':
        setRobotInfo(data.info);
        break;
      case 'state':
        if (data.motors) setMotorState(data.motors);
        if (data.head) setHeadPose(data.head);
        if (data.antennas) setAntennas(data.antennas);
        if (data.body_yaw !== undefined) setBodyYaw(data.body_yaw);
        break;
      case 'error':
        setLastError(data.message);
        break;
    }
  };

  const sendCommand = useCallback((command) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(command));
    }
  }, []);

  // Motor controls
  const enableMotors = () => sendCommand({ type: 'enable_motors' });
  const disableMotors = () => sendCommand({ type: 'disable_motors' });
  const toggleGravityComp = () => sendCommand({ type: 'toggle_gravity_compensation' });

  // Movement controls
  const updateHeadPose = (newPose) => {
    setHeadPose(newPose);
    sendCommand({
      type: 'set_head_pose',
      pose: newPose,
      duration: 0.5
    });
  };

  const updateAntennas = (newAntennas) => {
    setAntennas(newAntennas);
    sendCommand({
      type: 'set_antennas',
      antennas: newAntennas
    });
  };

  const updateBodyYaw = (yaw) => {
    setBodyYaw(yaw);
    sendCommand({
      type: 'set_body_yaw',
      yaw: yaw
    });
  };

  // Expression presets
  const playExpression = (expression) => {
    setCurrentExpression(expression);
    sendCommand({
      type: 'play_expression',
      expression: expression
    });
  };

  // Behaviors
  const wakeUp = () => sendCommand({ type: 'wake_up' });
  const goToSleep = () => sendCommand({ type: 'goto_sleep' });
  const lookAtCamera = () => sendCommand({ type: 'look_at_camera' });

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (robotSettings.autoConnect) {
      connectToRobot();
    }
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="tab-panel">
      <div className="tab-header">
        <h2>Robot Control</h2>
        <p>Connect to and control Reachy Mini robot for embodied AI interactions.</p>
      </div>

      <div className="tab-body">
        <div className="grid-2">
          {/* Connection Panel */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                <RobotIcon /> Connection
              </span>
              <ConnectionBadge status={robotStatus} />
            </div>

            <div className="form-group">
              <label className="form-label">Robot Host</label>
              <input
                type="text"
                className="form-input"
                value={robotSettings.host}
                onChange={(e) => updateRobotSettings('host', e.target.value)}
                placeholder="localhost"
                disabled={robotStatus === CONNECTION_STATES.CONNECTED}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Bridge Port</label>
              <input
                type="number"
                className="form-input"
                value={robotSettings.port}
                onChange={(e) => updateRobotSettings('port', parseInt(e.target.value))}
                placeholder="8001"
                disabled={robotStatus === CONNECTION_STATES.CONNECTED}
              />
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              {robotStatus !== CONNECTION_STATES.CONNECTED ? (
                <button
                  className="btn btn-primary"
                  onClick={connectToRobot}
                  disabled={robotStatus === CONNECTION_STATES.CONNECTING}
                >
                  {robotStatus === CONNECTION_STATES.CONNECTING ? 'Connecting...' : 'Connect'}
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={disconnectFromRobot}>
                  Disconnect
                </button>
              )}
            </div>

            {lastError && (
              <div className="status-error">
                <XIcon /> {lastError}
              </div>
            )}

            {robotInfo && (
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Robot</span>
                  <span className="metric-value">{robotInfo.name || 'Reachy Mini'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Version</span>
                  <span className="metric-value">{robotInfo.version || '-'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Mode</span>
                  <span className="metric-value">{robotInfo.mode || 'unknown'}</span>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '1rem' }}>
              <input
                type="checkbox"
                id="autoConnect"
                checked={robotSettings.autoConnect}
                onChange={(e) => updateRobotSettings('autoConnect', e.target.checked)}
              />
              <label htmlFor="autoConnect" style={{ fontSize: '0.875rem' }}>
                Auto-connect on startup
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
              <input
                type="checkbox"
                id="syncWithChat"
                checked={robotSettings.syncWithChat}
                onChange={(e) => updateRobotSettings('syncWithChat', e.target.checked)}
              />
              <label htmlFor="syncWithChat" style={{ fontSize: '0.875rem' }}>
                Sync expressions with chat responses
              </label>
            </div>
          </div>

          {/* Motor Control */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                <MotorIcon /> Motor Control
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              <button
                className={`btn ${motorState.enabled ? 'btn-primary' : 'btn-secondary'}`}
                onClick={motorState.enabled ? disableMotors : enableMotors}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
              >
                {motorState.enabled ? 'Motors ON' : 'Motors OFF'}
              </button>
              <button
                className={`btn ${motorState.gravityCompensation ? 'btn-primary' : 'btn-secondary'}`}
                onClick={toggleGravityComp}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
              >
                Gravity Comp: {motorState.gravityCompensation ? 'ON' : 'OFF'}
              </button>
            </div>

            <div className="card-header" style={{ marginTop: '1rem' }}>
              <span className="card-title">Quick Actions</span>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button
                className="btn btn-secondary"
                onClick={wakeUp}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
              >
                Wake Up
              </button>
              <button
                className="btn btn-secondary"
                onClick={goToSleep}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
              >
                Go to Sleep
              </button>
              <button
                className="btn btn-secondary"
                onClick={lookAtCamera}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
              >
                Look at Camera
              </button>
            </div>
          </div>
        </div>

        {/* Head Pose Control */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <HeadIcon /> Head Pose
            </span>
          </div>

          <div className="grid-3">
            {/* Position */}
            <div>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Position (mm)
              </h4>
              {['x', 'y', 'z'].map(axis => (
                <div key={axis} className="slider-group">
                  <div className="slider-header">
                    <span className="slider-label">{axis.toUpperCase()}</span>
                    <span className="slider-value">{headPose[axis]}mm</span>
                  </div>
                  <input
                    type="range"
                    className="slider-input"
                    min="-50"
                    max="50"
                    value={headPose[axis]}
                    onChange={(e) => updateHeadPose({ ...headPose, [axis]: parseInt(e.target.value) })}
                    disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                  />
                </div>
              ))}
            </div>

            {/* Rotation */}
            <div>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Rotation (degrees)
              </h4>
              {['roll', 'pitch', 'yaw'].map(axis => (
                <div key={axis} className="slider-group">
                  <div className="slider-header">
                    <span className="slider-label" style={{ textTransform: 'capitalize' }}>{axis}</span>
                    <span className="slider-value">{headPose[axis]}°</span>
                  </div>
                  <input
                    type="range"
                    className="slider-input"
                    min="-45"
                    max="45"
                    value={headPose[axis]}
                    onChange={(e) => updateHeadPose({ ...headPose, [axis]: parseInt(e.target.value) })}
                    disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                  />
                </div>
              ))}
            </div>

            {/* Antennas & Body */}
            <div>
              <h4 style={{ marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Antennas & Base
              </h4>
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Left Antenna</span>
                  <span className="slider-value">{Math.round(antennas.left * 57.3)}°</span>
                </div>
                <input
                  type="range"
                  className="slider-input"
                  min="-1.5"
                  max="1.5"
                  step="0.1"
                  value={antennas.left}
                  onChange={(e) => updateAntennas({ ...antennas, left: parseFloat(e.target.value) })}
                  disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                />
              </div>
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Right Antenna</span>
                  <span className="slider-value">{Math.round(antennas.right * 57.3)}°</span>
                </div>
                <input
                  type="range"
                  className="slider-input"
                  min="-1.5"
                  max="1.5"
                  step="0.1"
                  value={antennas.right}
                  onChange={(e) => updateAntennas({ ...antennas, right: parseFloat(e.target.value) })}
                  disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                />
              </div>
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-label">Body Yaw</span>
                  <span className="slider-value">{bodyYaw}°</span>
                </div>
                <input
                  type="range"
                  className="slider-input"
                  min="-180"
                  max="180"
                  value={bodyYaw}
                  onChange={(e) => updateBodyYaw(parseInt(e.target.value))}
                  disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                />
              </div>
            </div>
          </div>

          <button
            className="btn btn-secondary"
            onClick={() => {
              const neutral = { x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0 };
              updateHeadPose(neutral);
              updateAntennas({ left: 0, right: 0 });
              updateBodyYaw(0);
            }}
            disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
            style={{ marginTop: '1rem' }}
          >
            Reset to Neutral
          </button>
        </div>

        {/* Expressions */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <ExpressionIcon /> Expressions
            </span>
            <span className="tag">{currentExpression}</span>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {EXPRESSIONS.map(expr => (
              <button
                key={expr.id}
                className={`btn ${currentExpression === expr.id ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => playExpression(expr.id)}
                disabled={robotStatus !== CONNECTION_STATES.CONNECTED}
                title={expr.description}
              >
                {expr.emoji} {expr.name}
              </button>
            ))}
          </div>

          <p className="form-hint" style={{ marginTop: '1rem' }}>
            Expressions are pre-programmed animations combining head movement, antenna positions, and optional sounds.
          </p>
        </div>

        {/* Setup Instructions */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <InfoIcon /> Setup Instructions
            </span>
          </div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>1. Install the robot bridge:</strong>
            </p>
            <pre style={{
              background: 'var(--bg-tertiary)',
              padding: '0.75rem',
              borderRadius: '0.375rem',
              marginBottom: '1rem',
              overflow: 'auto'
            }}>
{`cd phase0/backend
pip install reachy-mini
python robot_bridge.py`}
            </pre>
            <p style={{ marginBottom: '0.75rem' }}>
              <strong>2. Connect your Reachy Mini:</strong>
            </p>
            <ul style={{ marginLeft: '1.5rem', marginBottom: '0.75rem' }}>
              <li><strong>Lite:</strong> Connect via USB-C</li>
              <li><strong>Wireless:</strong> Ensure WiFi is connected</li>
              <li><strong>Simulation:</strong> Start with <code>--sim</code> flag</li>
            </ul>
            <p>
              <strong>3. Enable "Sync with chat"</strong> to have NIVEK control the robot during conversations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Expression presets
const EXPRESSIONS = [
  { id: 'neutral', name: 'Neutral', emoji: '😐', description: 'Default neutral position' },
  { id: 'happy', name: 'Happy', emoji: '😊', description: 'Antenna wiggle with slight tilt' },
  { id: 'curious', name: 'Curious', emoji: '🤔', description: 'Head tilt with antenna perk' },
  { id: 'thinking', name: 'Thinking', emoji: '💭', description: 'Look up and to the side' },
  { id: 'listening', name: 'Listening', emoji: '👂', description: 'Lean forward attentively' },
  { id: 'nod', name: 'Nod', emoji: '👍', description: 'Affirmative nod' },
  { id: 'shake', name: 'Shake', emoji: '👎', description: 'Negative head shake' },
  { id: 'surprise', name: 'Surprised', emoji: '😲', description: 'Quick back motion, antennas up' },
  { id: 'sad', name: 'Sad', emoji: '😢', description: 'Droopy antennas, head down' },
  { id: 'excited', name: 'Excited', emoji: '🎉', description: 'Bouncy movement, fast antenna wiggle' },
];

// Connection status badge
function ConnectionBadge({ status }) {
  const colors = {
    [CONNECTION_STATES.CONNECTED]: 'var(--success)',
    [CONNECTION_STATES.CONNECTING]: 'var(--warning)',
    [CONNECTION_STATES.DISCONNECTED]: 'var(--text-muted)',
    [CONNECTION_STATES.ERROR]: 'var(--error)'
  };

  return (
    <span className="tag" style={{ backgroundColor: colors[status], color: 'white' }}>
      {status}
    </span>
  );
}

// Icons
function RobotIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="11" width="18" height="10" rx="2"/>
      <circle cx="12" cy="5" r="3"/>
      <line x1="12" y1="8" x2="12" y2="11"/>
      <line x1="8" y1="16" x2="8" y2="16"/>
      <line x1="16" y1="16" x2="16" y2="16"/>
    </svg>
  );
}

function MotorIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3"/>
      <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
    </svg>
  );
}

function HeadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="10" r="7"/>
      <path d="M12 17v4"/>
      <path d="M8 21h8"/>
      <circle cx="9" cy="9" r="1"/>
      <circle cx="15" cy="9" r="1"/>
    </svg>
  );
}

function ExpressionIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
      <line x1="9" y1="9" x2="9.01" y2="9"/>
      <line x1="15" y1="9" x2="15.01" y2="9"/>
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

function XIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" y1="9" x2="9" y2="15"/>
      <line x1="9" y1="9" x2="15" y2="15"/>
    </svg>
  );
}
