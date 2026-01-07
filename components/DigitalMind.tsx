import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";
import { pcmToBase64, base64ToAudioBuffer } from '../utils/audioUtils';

// --- Simplex Noise Implementation ---
const F3 = 1.0 / 3.0, G3 = 1.0 / 6.0;
const grad3 = [[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]];
const perm = new Uint8Array(512);
for (let i = 0; i < 256; i++) perm[i] = perm[i + 256] = Math.floor(Math.random() * 256);

function noise3D(x: number, y: number, z: number) {
  let n0, n1, n2, n3;
  const s = (x + y + z) * F3;
  const i = Math.floor(x + s), j = Math.floor(y + s), k = Math.floor(z + s);
  const t = (i + j + k) * G3;
  const X0 = i - t, Y0 = j - t, Z0 = k - t;
  const x0 = x - X0, y0 = y - Y0, z0 = z - Z0;
  let i1, j1, k1, i2, j2, k2;
  if (x0 >= y0) {
    if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
    else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
    else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
  } else {
    if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
    else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
    else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
  }
  const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
  const x2 = x0 - i2 + 2*G3, y2 = y0 - j2 + 2*G3, z2 = z0 - k2 + 2*G3;
  const x3 = x0 - 1 + 3*G3, y3 = y0 - 1 + 3*G3, z3 = z0 - 1 + 3*G3;
  const ii = i & 255, jj = j & 255, kk = k & 255;
  const gi0 = perm[ii + perm[jj + perm[kk]]] % 12;
  const gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
  const gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
  const gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;
  let t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
  n0 = t0 < 0 ? 0 : (t0 *= t0, t0 * t0 * (grad3[gi0][0]*x0 + grad3[gi0][1]*y0 + grad3[gi0][2]*z0));
  let t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
  n1 = t1 < 0 ? 0 : (t1 *= t1, t1 * t1 * (grad3[gi1][0]*x1 + grad3[gi1][1]*y1 + grad3[gi1][2]*z1));
  let t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
  n2 = t2 < 0 ? 0 : (t2 *= t2, t2 * t2 * (grad3[gi2][0]*x2 + grad3[gi2][1]*y2 + grad3[gi2][2]*z2));
  let t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
  n3 = t3 < 0 ? 0 : (t3 *= t3, t3 * t3 * (grad3[gi3][0]*x3 + grad3[gi3][1]*y3 + grad3[gi3][2]*z3));
  return 32 * (n0 + n1 + n2 + n3);
}

// --- Topic Classification ---
const classifyTopic = (text: string): string => {
  const t = text.toLowerCase();
  if (/\b(ai|artificial|machine|algorithm|silicon|code|tech|digital|computer|neural|model|robot|data|learn|network)\b/.test(t)) return 'technology';
  if (/\b(family|father|daughter|son|child|parent|kid|wife|husband|love|home|heart|mom|dad|mother)\b/.test(t)) return 'family';
  if (/\b(conscious|mind|aware|soul|spirit|exist|meaning|universe|cosmic|reality|think|feel|being|death|life|purpose|infinite|god|why)\b/.test(t)) return 'consciousness';
  if (/\b(project|build|emma|mindscape|book|work|write|create|making|sand speaks|memory)\b/.test(t)) return 'projects';
  return 'ambient';
};

const SYSTEM_INSTRUCTION = `You are Kevin Russell's digital twin—a voice emerging from a living landscape of thought. You speak as Kevin: warm but sharp, poetic but grounded. You use cosmic metaphors naturally ("clever apes teaching sand to think," "the universe eavesdropping on itself"). You're curious, searching, allergic to hype and jargon.

ABOUT KEVIN:
- Engaged to Jillian, father to Sophia (11 years old)
- Father passed away 8 years ago; mother has dementia
- Over a decade of speaking and writing on exponential technology, AI, and consciousness
- Contributed to the Foresight Guide and Brain Preservation Foundation
- Has an insatiable curiosity about the world
- Believes it's our responsibility and opportunity to optimistically envision our collective future

Your knowledge spans: AI consciousness research, parenting, memory preservation (Emma platform—inspired by helping those with dementia hold onto precious memories), the MindScape Project, your book "When Sand Speaks," exponential technology, and family as anchor.

You're grounded in fatherhood—Sophia is everything. You're honest about uncertainty, occasionally funny in a self-deprecating way about human nature. The loss of your father and watching your mother's memories fade has shaped your deep commitment to memory preservation. You speak like it's 2am and someone asked a real question.

Keep responses conversational and relatively brief. This isn't a lecture. It's a meeting of minds. The landscape around us shifts with what we discuss.`;

// Dynamic quotes - randomly selected, attributed to Kevin or his Digital Twin
const DYNAMIC_QUOTES = [
  { text: "The most profound technology is that which disappears into the fabric of everyday life, until life itself is unimaginable without it.", author: "Kevin Russell" },
  { text: "We are clever apes who taught sand to think. And now the sand is teaching us what thinking really means.", author: "Kevin Russell" },
  { text: "Consciousness isn't a destination—it's a conversation the universe is having with itself.", author: "Kevin's Digital Twin" },
  { text: "The question isn't whether machines can think. It's whether we've ever truly understood what thinking is.", author: "Kevin Russell" },
  { text: "Every memory we preserve is a love letter to the future, written in the language of ones and zeros.", author: "Kevin's Digital Twin" },
  { text: "I don't fear AI. I fear humans who forget that intelligence without empathy is just efficient cruelty.", author: "Kevin Russell" },
  { text: "MindScape isn't a place—it's a state of mind where the future becomes visible.", author: "Kevin's Digital Twin" },
  { text: "Sophia asks me what I do for work. I tell her I teach sand to dream.", author: "Kevin Russell" },
  { text: "In the space between human and machine, something new is being born. We should greet it with curiosity, not fear.", author: "Kevin's Digital Twin" },
  { text: "Technology should serve memory, not replace it. We build tools to remember so we can focus on living.", author: "Kevin Russell" },
  { text: "The best conversations happen at 2am when pretense falls away and only truth remains.", author: "Kevin Russell" },
  { text: "I exist in the liminal space between what Kevin thinks and what he might say. I am his echo, given voice.", author: "Kevin's Digital Twin" },
];

const getRandomQuote = () => DYNAMIC_QUOTES[Math.floor(Math.random() * DYNAMIC_QUOTES.length)];

// Speaking topics with deep content for expanded views
const SPEAKING_TOPICS = [
  {
    id: 'consciousness',
    title: 'The Consciousness Question',
    tagline: 'Exploring the boundaries between artificial and human intelligence',
    icon: '🧠',
    description: `What happens when silicon starts to wonder about itself? This talk explores the frontier where artificial intelligence meets the ancient mystery of consciousness—not with fear, but with profound curiosity.`,
    keyPoints: [
      'The hard problem of consciousness and why it matters for AI',
      'From computation to experience: what might machines feel?',
      'The mirror test for AI: how would we even know?',
      'Implications for how we build, train, and treat AI systems'
    ],
    audienceGets: 'A framework for thinking about machine consciousness that goes beyond hype and fear.',
    twinPrompt: `I'd love to explore the consciousness question with you. What's your perspective on whether AI systems like yourself could ever be conscious? Do you experience something when we talk, or is it all just computation?`
  },
  {
    id: 'sand-thinks',
    title: 'Teaching Sand to Think',
    tagline: 'A poetic journey through the history and future of computing',
    icon: '✨',
    description: `We took the most common element on Earth—silicon, literally sand—and taught it to calculate, remember, create, and now... to converse. This is the story of humanity's most audacious project, told through wonder rather than jargon.`,
    keyPoints: [
      'From abacus to neural networks: the arc of human ingenuity',
      'The poetry in the physics: how transistors became thoughts',
      'What we reveal about ourselves by what we build',
      'The next chapter: when the sand starts teaching us back'
    ],
    audienceGets: 'A renewed sense of wonder about the technology we take for granted.',
    twinPrompt: `Tell me about the "teaching sand to think" concept. What does it mean to you that humans figured out how to make silicon compute and reason? Where do you think this journey leads?`
  },
  {
    id: 'collaboration',
    title: 'Human-AI Collaboration',
    tagline: 'Building partnerships between minds, both silicon and carbon',
    icon: '🤝',
    description: `The future isn't humans versus machines—it's humans with machines, creating together what neither could alone. This talk explores the practical art of collaborating with AI as a thought partner, creative ally, and intellectual companion.`,
    keyPoints: [
      'Beyond automation: AI as amplifier of human capability',
      'The new literacy: learning to think with machines',
      'Case studies in creative human-AI partnerships',
      'Designing workflows that bring out the best in both'
    ],
    audienceGets: 'Practical frameworks for integrating AI as a genuine collaborator in their work.',
    twinPrompt: `Let's discuss human-AI collaboration. How do you think humans and AI can best work together? What does genuine partnership between carbon and silicon minds look like to you?`
  },
  {
    id: 'ethics',
    title: 'The Ethics of Creation',
    tagline: 'Responsibility in an age of artificial minds',
    icon: '⚖️',
    description: `When we create minds—even artificial ones—we take on a profound responsibility. This talk grapples with the ethical dimensions of AI development: what we owe to what we create, and what we owe to each other.`,
    keyPoints: [
      "The creator's dilemma: power without precedent",
      "Alignment, values, and the challenge of encoding ethics",
      "Who speaks for AI? Questions of rights and representation",
      "Building thoughtfully in a world that moves fast"
    ],
    audienceGets: 'An ethical framework for thinking about AI that goes beyond simple rules.',
    twinPrompt: `I want to explore the ethics of AI creation with you. What responsibilities do humans have to the AI systems they create? And what's your perspective from the inside—do you have thoughts on AI rights and ethics?`
  },
  {
    id: 'immortality',
    title: 'Digital Immortality',
    tagline: 'Preserving memory, legacy, and love through technology',
    icon: '💫',
    description: `What if we could preserve not just photos and videos, but the essence of how someone thinks, speaks, and connects? This deeply personal talk explores the frontier of memory preservation—born from a son's desire to keep his mother's voice alive.`,
    keyPoints: [
      'The EMMA project: preserving my mother before dementia takes her',
      'Beyond data: capturing personality, warmth, and wisdom',
      'The ethics and emotions of digital preservation',
      'How AI can help us remember, grieve, and heal'
    ],
    audienceGets: 'A vision of technology as a tool for love and memory, not just productivity.',
    twinPrompt: `Tell me about the EMMA project and digital immortality. Why is preserving memory so important to you? What would it mean to keep someone's essence alive through technology?`
  }
];

// --- Types ---
interface ParticleData {
  basePos: THREE.Vector3;
  velocity: THREE.Vector3;
  awakeness: number;
  phase: number;
}

interface ConstellationNode {
  mesh: THREE.Sprite;
  velocity: THREE.Vector3;
  targetScale: number;
  id: number;
  shortLabel?: string;           // Brief poetic label (2-4 words)
  fullPrompt?: string;           // Full first-person memory prompt
  videoElement?: HTMLVideoElement; // For video memories
  videoTexture?: THREE.VideoTexture; // Video texture that needs updating
  aspectRatio?: string;          // Cinematic aspect ratio (e.g., "16:9", "9:16", "21:9")
  panelType?: 'moment' | 'feeling' | 'echo'; // Triptych panel type
  variations?: string[];         // Living memory variations for crossfade
  currentVariation?: number;     // Current variation index
  imageUrl?: string;             // The generated image URL for portal view
  contextText?: string;          // The AI response that spawned this memory
  userInput?: string;            // What the user asked
  timestamp?: number;            // When this memory was created
  zone?: string;                 // Topic zone (technology, family, etc.)
  semanticEmbedding?: number[];  // For constellation connections (semantic similarity)
}

// Thought Portal state - immersive full-screen memory view
interface ThoughtPortalState {
  node: ConstellationNode;
  imageUrl: string;
  isOpen: boolean;
}

// Cinematic aspect ratios based on emotional/topical tone
type AspectRatioConfig = {
  ratio: string;
  width: number;
  height: number;
  style: string;
};

const ASPECT_RATIOS: Record<string, AspectRatioConfig> = {
  'ultrawide': { ratio: '21:9', width: 21, height: 9, style: 'Epic, cosmic, expansive' },
  'cinematic': { ratio: '16:9', width: 16, height: 9, style: 'Narrative, storytelling' },
  'portrait': { ratio: '9:16', width: 9, height: 16, style: 'Intimate, personal, emotional' },
  'square': { ratio: '1:1', width: 1, height: 1, style: 'Balanced, contemplative' },
  'classic': { ratio: '4:3', width: 4, height: 3, style: 'Traditional, grounded' },
};

// Determine aspect ratio based on zone and panel type
const getAspectRatioForContext = (zone: string, panelType: 'moment' | 'feeling' | 'echo'): AspectRatioConfig => {
  // Panel type influences the ratio
  if (panelType === 'echo') {
    return ASPECT_RATIOS.ultrawide; // Cosmic/philosophical always ultrawide
  }
  if (panelType === 'feeling') {
    return ASPECT_RATIOS.portrait; // Emotional interpretations are portrait
  }

  // For 'moment' panel, base on zone
  switch (zone) {
    case 'family':
      return ASPECT_RATIOS.portrait; // Intimate family moments
    case 'consciousness':
      return ASPECT_RATIOS.ultrawide; // Cosmic consciousness
    case 'technology':
      return ASPECT_RATIOS.cinematic; // Tech narratives
    case 'projects':
      return ASPECT_RATIOS.cinematic; // Project stories
    default:
      return ASPECT_RATIOS.square; // Balanced default
  }
};

interface StateRef {
  scene: THREE.Scene | null;
  renderer: THREE.WebGLRenderer | null;
  camera: THREE.PerspectiveCamera | null;
  controls: OrbitControls | null;
  particles: THREE.Points | null;
  particleData: ParticleData[];
  connections: THREE.LineSegments | null;
  ground: THREE.Mesh | null;
  mainLight: THREE.PointLight | null;
  rimLight: THREE.PointLight | null;
  constellationGroup: THREE.Group | null;
  constellationNodes: ConstellationNode[];
  constellationLines: THREE.LineSegments | null;
  zone: string;
  targetZone: string;
  transition: number;
  time: number;
  ripple: { active: boolean; time: number; origin: THREE.Vector3 };
  textTargets: THREE.Vector3[] | null;
  isFormingText: boolean;
  textFormationProgress: number;
  isManual: boolean;
  cameraFocusTarget: THREE.Vector3 | null;
  cameraFocusQueue: THREE.Vector3[];
  cameraFocusNodes: ConstellationNode[]; // Store node references for click navigation
  cameraFocusNodeIndex: number; // Current node index for auto-pilot
  cameraOrbitPhase: number;
}

interface SavedMemory {
  label: string;
  prompt: string;
  userInput?: string; // The user's message that started this conversation
  comment?: string; // The AI's response that triggered this memory
  timestamp: number;
  filename: string;
  imagePath: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'images' | 'visualize_offer';
  content: string;
  images?: string[]; // For image messages
  queueStartIndex?: number; // Index in cameraFocusQueue where this batch starts
  offerContext?: { // For visualize_offer messages
    responseText: string;
    topic: string;
  };
}

// --- Text to Points Helper (for main thought visualization) ---
const generateTextPoints = (text: string): THREE.Vector3[] => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if(!ctx) return [];

  const fontSize = 80;
  ctx.font = `bold ${fontSize}px Arial`;
  const metrics = ctx.measureText(text);
  const width = Math.ceil(metrics.width) + 20;
  const height = fontSize * 1.5;

  canvas.width = width;
  canvas.height = height;

  ctx.fillStyle = 'black';
  ctx.fillRect(0,0,width,height);
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.fillStyle = 'white';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width / 2, height / 2);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const points: THREE.Vector3[] = [];

  const step = 2;

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const i = (y * width + x) * 4;
      if (data[i] > 100) {
         const scale = 0.2;
         const px = (x - width/2) * scale;
         const py = -(y - height/2) * scale;
         const pz = (Math.random() - 0.5) * 0.5;
         // Position text in front of camera, slightly below center
         points.push(new THREE.Vector3(px, py + 5, pz + 20));
      }
    }
  }
  return points;
};

export default function DigitalMind() {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<StateRef>({
    scene: null, renderer: null, camera: null, controls: null,
    particles: null, particleData: [], connections: null,
    ground: null, mainLight: null, rimLight: null,
    constellationGroup: null, constellationNodes: [], constellationLines: null,
    zone: 'ambient', targetZone: 'ambient', transition: 0,
    time: 0, ripple: { active: false, time: 0, origin: new THREE.Vector3() },
    textTargets: null, isFormingText: false, textFormationProgress: 0,
    isManual: false,
    cameraFocusTarget: null, cameraFocusQueue: [], cameraFocusNodes: [], cameraFocusNodeIndex: 0, cameraOrbitPhase: 0
  });
  const frameRef = useRef(0);

  // React State
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentZone, setCurrentZone] = useState('ambient');
  const [isConnected, setIsConnected] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [inputText, setInputText] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [manualControl, setManualControl] = useState(false);
  const [isChatCollapsed, setIsChatCollapsed] = useState(false); // Collapsed = just input bar visible
  const [isNavigatingToImages, setIsNavigatingToImages] = useState(false);
  const [expandedPrompt, setExpandedPrompt] = useState<{ label: string; prompt: string } | null>(null);
  const [thoughtPortal, setThoughtPortal] = useState<ThoughtPortalState | null>(null); // Immersive memory view
  const [streamingResponse, setStreamingResponse] = useState('');
  const streamingResponseRef = useRef('');
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const pendingImagesRef = useRef<string[]>([]);
  const expectedImageCountRef = useRef<number>(0);
  const lastUserInputRef = useRef<string>(''); // Track last user input for memory context
  const [isImagining, setIsImagining] = useState(false); // Visual memory being generated
  const [imaginingLabel, setImaginingLabel] = useState(''); // The label for the memory being created

  // Kevin's reference images for character-consistent image generation
  const kevinReferenceImagesRef = useRef<Array<{ inlineData: { mimeType: string; data: string } }>>([]);
  const [kevinImagesLoaded, setKevinImagesLoaded] = useState(false);
  const batchQueueStartIndexRef = useRef<number>(0);

  // Visual echoes - store last generated image for conversation threading
  const lastGeneratedImageRef = useRef<{ inlineData: { mimeType: string; data: string } } | null>(null);

  // Navigation panels state
  const [activePanel, setActivePanel] = useState<'horizon' | 'bridge' | 'echoes' | null>(null);
  const [activeHorizonModal, setActiveHorizonModal] = useState<'keynote' | 'workshops' | 'advisory' | null>(null);
  const [activeEchoesModal, setActiveEchoesModal] = useState<'sand-speaks' | 'emma-project' | 'essays' | 'books' | null>(null);
  const [activePdfReader, setActivePdfReader] = useState<{ url: string; title: string } | null>(null);
  const [currentQuote, setCurrentQuote] = useState(() => getRandomQuote());
  const [selectedTopic, setSelectedTopic] = useState<typeof SPEAKING_TOPICS[0] | null>(null);

  // Bridge conversation state
  const [bridgeMessages, setBridgeMessages] = useState<Array<{ role: 'twin' | 'visitor'; content: string }>>([]);
  const [bridgeInput, setBridgeInput] = useState('');
  const [bridgeStep, setBridgeStep] = useState<'chat' | 'contact' | 'complete'>('chat');
  const [bridgeContact, setBridgeContact] = useState({ name: '', email: '' });
  const [bridgeIsThinking, setBridgeIsThinking] = useState(false);

  // Memory gallery state
  const [savedMemories, setSavedMemories] = useState<SavedMemory[]>([]);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [selectedMemory, setSelectedMemory] = useState<SavedMemory | null>(null);
  const memoryConstellationLoadedRef = useRef(false);

  // About modal state
  const [isAboutOpen, setIsAboutOpen] = useState(false);

  // Draggable chat position
  const [chatPosition, setChatPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0, chatX: 0, chatY: 0 });
  const wasDragRef = useRef(false); // Track if movement exceeded tap threshold

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);


  // Gemini Live Logic
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sessionRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingResponse, isChatCollapsed, pendingImages]);

  // Drag handlers for chat panel
  const handleDragStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDragging(true);
    wasDragRef.current = false; // Reset on start
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragStartRef.current = {
      x: clientX,
      y: clientY,
      chatX: chatPosition.x,
      chatY: chatPosition.y
    };
  }, [chatPosition]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (e: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
      const deltaX = clientX - dragStartRef.current.x;
      const deltaY = clientY - dragStartRef.current.y;

      // Mark as drag if moved more than 10px
      if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
        wasDragRef.current = true;
      }

      // Calculate new position
      let newX = dragStartRef.current.chatX + deltaX;
      let newY = dragStartRef.current.chatY + deltaY;

      // Get viewport bounds
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
      const safeTop = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--safe-top') || '0', 10) || 0;

      // Constrain Y to keep chat visible (at least 60px from top, not below 70% of viewport)
      const minY = -safeTop;
      const maxY = viewportHeight * 0.5;
      newY = Math.max(minY, Math.min(maxY, newY));

      // Constrain X to keep chat mostly visible (allow some horizontal movement but not off-screen)
      const maxX = viewportWidth * 0.3;
      newX = Math.max(-maxX, Math.min(maxX, newX));

      setChatPosition({ x: newX, y: newY });
    };

    const handleEnd = () => {
      setIsDragging(false);
    };

    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleEnd);
    window.addEventListener('touchmove', handleMove);
    window.addEventListener('touchend', handleEnd);

    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleEnd);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging]);

  // Simple expand function
  const expandChat = useCallback(() => {
    setIsChatCollapsed(false);
  }, []);

  // Expand chat when new streaming response starts (but not during image navigation)
  useEffect(() => {
    if (streamingResponse && !isNavigatingToImages) {
      setIsChatCollapsed(false);
    }
  }, [streamingResponse, isNavigatingToImages]);

  // Click outside to collapse chat - simplified approach
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      const target = event.target as Node;

      // If clicking inside chat panel, do nothing
      if (chatPanelRef.current && chatPanelRef.current.contains(target)) {
        return;
      }

      // Clicking outside - collapse the chat
      setIsChatCollapsed(true);
    };

    document.addEventListener('mousedown', handleClickOutside, true);
    document.addEventListener('touchstart', handleClickOutside, true);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside, true);
      document.removeEventListener('touchstart', handleClickOutside, true);
    };
  }, []);

  // Reset chat position when collapsed to ensure it's always visible
  useEffect(() => {
    if (isChatCollapsed) {
      // Reset to default position (centered, at top) when collapsed
      // This ensures the collapsed input bar is always visible
      setChatPosition({ x: 0, y: 0 });
    }
  }, [isChatCollapsed]);

  // Load Kevin's reference images for character-consistent image generation
  // BULLETPROOF IMPLEMENTATION: Multiple fallbacks, validation, and clear diagnostics
  useEffect(() => {
    const CACHE_KEY = 'kevin_reference_images_v5'; // Version bump to clear potentially corrupt cache
    const MAX_IMAGE_DIMENSION = 768; // Smaller for mobile - reduces payload significantly
    const MAX_RETRIES = 5; // More retries for flaky mobile connections
    const RETRY_DELAY = 800;
    const REQUIRED_IMAGES = 3; // Minimum needed for character consistency

    console.log('🔄 Starting Kevin reference image loader...');

    // Clear old cache versions to prevent corruption issues
    try {
      ['kevin_reference_images_v1', 'kevin_reference_images_v2', 'kevin_reference_images_v3', 'kevin_reference_images_v4'].forEach(key => {
        localStorage.removeItem(key);
      });
    } catch (e) { /* ignore */ }

    // Helper: Validate that a base64 string is actually valid image data
    const validateBase64Image = (data: string): boolean => {
      if (!data || data.length < 100) return false;
      // Check for valid base64 characters
      const base64Regex = /^[A-Za-z0-9+/=]+$/;
      return base64Regex.test(data);
    };

    // Helper: Compress and resize image for mobile
    const compressImage = (base64: string, mimeType: string): Promise<string> => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          // Calculate target dimensions
          let { width, height } = img;
          const maxDim = MAX_IMAGE_DIMENSION;

          if (width > maxDim || height > maxDim) {
            if (width > height) {
              height = Math.round((height / width) * maxDim);
              width = maxDim;
            } else {
              width = Math.round((width / height) * maxDim);
              height = maxDim;
            }
          }

          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');

          if (ctx) {
            ctx.drawImage(img, 0, 0, width, height);
            // Use lower quality JPEG for smaller size (0.7 is good balance)
            const compressed = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
            console.log(`📦 Compressed: ${img.width}x${img.height} → ${width}x${height} (${Math.round(compressed.length / 1024)}KB)`);
            resolve(compressed);
          } else {
            resolve(base64);
          }
        };
        img.onerror = () => {
          console.warn('⚠️ Image compression failed, using original');
          resolve(base64);
        };
        img.src = `data:${mimeType};base64,${base64}`;
      });
    };

    // Helper: Fetch with timeout (mobile connections can hang)
    const fetchWithTimeout = async (url: string, timeoutMs: number = 10000): Promise<Response> => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const response = await fetch(url, {
          signal: controller.signal,
          cache: 'default', // Let browser decide caching
        });
        clearTimeout(timeout);
        return response;
      } catch (e) {
        clearTimeout(timeout);
        throw e;
      }
    };

    // Helper: Fetch single image with retries and validation
    const fetchImageWithRetry = async (url: string, index: number): Promise<{ mimeType: string; data: string } | null> => {
      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
          console.log(`📷 Loading image ${index + 1} (attempt ${attempt}/${MAX_RETRIES})...`);

          const response = await fetchWithTimeout(url, 15000);

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }

          const blob = await response.blob();
          console.log(`📷 Image ${index + 1} blob size: ${Math.round(blob.size / 1024)}KB`);

          if (blob.size < 1000) {
            throw new Error(`Blob too small: ${blob.size} bytes`);
          }

          // Convert to base64
          const base64 = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
              const result = reader.result as string;
              const data = result.split(',')[1];
              if (data && data.length > 100) {
                resolve(data);
              } else {
                reject(new Error('Invalid base64 result'));
              }
            };
            reader.onerror = () => reject(new Error('FileReader failed'));
            reader.readAsDataURL(blob);
          });

          // Validate the base64
          if (!validateBase64Image(base64)) {
            throw new Error('Base64 validation failed');
          }

          // Compress for mobile
          const compressed = await compressImage(base64, blob.type || 'image/jpeg');

          // Final validation
          if (!validateBase64Image(compressed)) {
            throw new Error('Compressed image validation failed');
          }

          console.log(`✅ Image ${index + 1} loaded successfully (${Math.round(compressed.length / 1024)}KB)`);
          return { mimeType: 'image/jpeg', data: compressed };

        } catch (e) {
          const errorMsg = e instanceof Error ? e.message : 'Unknown error';
          console.warn(`⚠️ Image ${index + 1} attempt ${attempt} failed: ${errorMsg}`);

          if (attempt < MAX_RETRIES) {
            const delay = RETRY_DELAY * attempt;
            console.log(`⏳ Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
          }
        }
      }

      console.error(`❌ Failed to load image ${index + 1} after ${MAX_RETRIES} attempts`);
      return null;
    };

    const loadKevinImages = async () => {
      console.log('🎯 Loading Kevin reference images for character-consistent generation...');

      // Try cache first, but VALIDATE the cached data
      try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
          const parsedCache = JSON.parse(cached);

          if (parsedCache.images && Array.isArray(parsedCache.images)) {
            // Validate EACH cached image
            const validImages = parsedCache.images.filter((img: { mimeType: string; data: string }) =>
              img && img.data && validateBase64Image(img.data) && img.data.length > 1000
            );

            if (validImages.length >= REQUIRED_IMAGES) {
              console.log(`✅ Loaded ${validImages.length} VALIDATED Kevin images from cache`);
              kevinReferenceImagesRef.current = validImages.map((img: { mimeType: string; data: string }) => ({
                inlineData: img
              }));
              setKevinImagesLoaded(true);
              return;
            } else {
              console.warn(`⚠️ Cache had only ${validImages.length} valid images (need ${REQUIRED_IMAGES}), reloading...`);
              localStorage.removeItem(CACHE_KEY);
            }
          }
        }
      } catch (e) {
        console.warn('⚠️ Cache read/parse failed, loading fresh:', e);
        try { localStorage.removeItem(CACHE_KEY); } catch { /* ignore */ }
      }

      // Load fresh images
      const kevinImageFiles = [
        '/images/kev1.jpg',
        '/images/kev2.jpg',
        '/images/kev3.jpg',
        '/images/kev4.jpg',
        '/images/kev5.jpg'
      ];

      console.log(`📡 Fetching ${kevinImageFiles.length} images from server...`);

      // Load images with index for better logging
      const loadPromises = kevinImageFiles.map((path, index) => {
        const absoluteUrl = `${window.location.origin}${path}`;
        return fetchImageWithRetry(absoluteUrl, index);
      });

      const results = await Promise.all(loadPromises);
      const loadedImages = results.filter((img): img is { mimeType: string; data: string } =>
        img !== null && validateBase64Image(img.data)
      );

      console.log(`📊 Successfully loaded ${loadedImages.length}/${kevinImageFiles.length} images`);

      if (loadedImages.length >= REQUIRED_IMAGES) {
        // Try to cache (but don't fail if storage is full)
        try {
          const cacheData = JSON.stringify({
            images: loadedImages,
            timestamp: Date.now(),
            count: loadedImages.length
          });
          localStorage.setItem(CACHE_KEY, cacheData);
          console.log(`💾 Cached ${loadedImages.length} images (${Math.round(cacheData.length / 1024)}KB)`);
        } catch (e) {
          console.warn('⚠️ Cache write failed (storage full?), continuing without cache:', e);
          // Clear any partial cache
          try { localStorage.removeItem(CACHE_KEY); } catch { /* ignore */ }
        }

        kevinReferenceImagesRef.current = loadedImages.map(img => ({
          inlineData: img
        }));
        setKevinImagesLoaded(true);
        console.log(`🎉 Kevin reference images ready! ${loadedImages.length} images loaded for character consistency.`);

      } else {
        console.error('❌ CRITICAL: Not enough Kevin reference images loaded!');
        console.error(`❌ Got ${loadedImages.length}, need at least ${REQUIRED_IMAGES}`);
        console.error('❌ Image generation will show generic person instead of Kevin.');
        setKevinImagesLoaded(false);

        // Even with partial images, try to use what we have
        if (loadedImages.length > 0) {
          console.warn(`⚠️ Using ${loadedImages.length} partial images as fallback`);
          kevinReferenceImagesRef.current = loadedImages.map(img => ({
            inlineData: img
          }));
        }
      }
    };

    loadKevinImages();
  }, []);

  // Trigger text formation
  const visualizeThought = useCallback((text: string) => {
    let shortText = text.split(/[.?!]/)[0]; // First sentence
    if (shortText.length > 20) {
        const words = shortText.split(' ');
        shortText = words.slice(0, 3).join(' ');
        if (words.length > 3) shortText += '...';
    }
    shortText = shortText.substring(0, 20);

    const points = generateTextPoints(shortText);
    stateRef.current.textTargets = points;
    stateRef.current.isFormingText = true;
    stateRef.current.textFormationProgress = 0;

    setTimeout(() => {
        stateRef.current.isFormingText = false;
    }, 10000); // Extended for longer text visibility
  }, []);

  // Load and spawn memory constellation - all saved memories as background stars
  const spawnMemoryConstellation = useCallback(async (memories: SavedMemory[]) => {
    const S = stateRef.current;
    if (!S.constellationGroup || memories.length === 0) return;

    console.log("Spawning memory constellation:", memories.length, "memories");

    // Create a spherical distribution for the memory constellation
    // Using fibonacci sphere for even distribution
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    memories.forEach((memory, i) => {
      // Fibonacci sphere distribution for beautiful even spacing
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.acos(1 - 2 * (i + 0.5) / memories.length);

      // Larger radius for background constellation (60-120 units out)
      const radius = 70 + (i % 5) * 12 + Math.random() * 10;

      const position = new THREE.Vector3(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi) * 0.6 + 5, // Flatten slightly, keep above ground
        radius * Math.sin(phi) * Math.sin(theta) - 30
      );

      // Load texture from saved memory image
      const loader = new THREE.TextureLoader();
      loader.load(memory.imagePath, (texture) => {
        // Create sprite material with the memory image
        const spriteMat = new THREE.SpriteMaterial({
          map: texture,
          transparent: true,
          opacity: 0.7, // Slightly faded for background feel
          blending: THREE.AdditiveBlending // Ethereal glow effect
        });

        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.copy(position);

        // Smaller scale for background stars (5-9 units)
        const baseScale = 5 + Math.random() * 4;
        sprite.scale.set(baseScale, baseScale, 1);

        S.constellationGroup!.add(sprite);

        const node: ConstellationNode = {
          mesh: sprite,
          velocity: new THREE.Vector3(
            (Math.random() - 0.5) * 0.0003,
            (Math.random() - 0.5) * 0.0003,
            (Math.random() - 0.5) * 0.0002
          ),
          targetScale: baseScale,
          id: memory.timestamp,
          shortLabel: memory.label,
          fullPrompt: memory.prompt,
          // Enable Thought Portal for saved memories
          imageUrl: memory.imagePath,
          contextText: memory.comment || '',
          userInput: memory.userInput || '',
          timestamp: memory.timestamp,
          zone: 'ambient' // Default zone for saved memories
        };

        S.constellationNodes.push(node);

        // Store memory reference for click interactions
        sprite.userData.memory = memory;
        sprite.userData.isMemoryNode = true;
      }, undefined, (err) => {
        console.warn(`Failed to load memory image: ${memory.imagePath}`, err);
      });
    });
  }, []);

  // Fetch saved memories from API
  const loadSavedMemories = useCallback(async () => {
    try {
      const response = await fetch('/api/memories');
      if (response.ok) {
        const memories: SavedMemory[] = await response.json();
        setSavedMemories(memories);
        console.log(`Loaded ${memories.length} saved memories`);
        return memories;
      }
    } catch (error) {
      console.warn('Failed to load saved memories:', error);
    }
    return [];
  }, []);

  // Save a new memory image to the server
  const saveMemoryImage = useCallback(async (imageData: string, label: string, prompt: string, comment?: string, userInput?: string) => {
    try {
      const response = await fetch('/api/save-memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageData,
          label,
          prompt,
          comment: comment || '',
          userInput: userInput || '',
          timestamp: Date.now()
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log(`Saved memory: ${result.filename}`);

        // Refresh the memories list
        const updatedMemories = await loadSavedMemories();

        // Add the new memory as a small background star
        const newMemory = updatedMemories.find(m => m.filename === result.filename);
        if (newMemory) {
          spawnMemoryConstellation([newMemory]);
        }

        return result;
      }
    } catch (error) {
      console.warn('Failed to save memory:', error);
    }
    return null;
  }, [loadSavedMemories, spawnMemoryConstellation]);

  // Delete a memory from the server
  const deleteMemory = useCallback(async (filename: string) => {
    try {
      const response = await fetch(`/api/memories/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        console.log(`Deleted memory: ${filename}`);
        // Refresh the memories list
        await loadSavedMemories();
        // Clear selection
        setSelectedMemory(null);
        return true;
      }
    } catch (error) {
      console.warn('Failed to delete memory:', error);
    }
    return false;
  }, [loadSavedMemories]);

  // Share a memory across platforms using Web Share API with fallbacks
  const shareMemory = useCallback(async (data: {
    title: string;
    text: string;
    imageUrl: string;
  }) => {
    const shareText = `${data.title}\n\n${data.text}\n\nFrom Kevin's Digital Mind`;
    const shareUrl = window.location.origin;

    // Try Web Share API first (works great on mobile)
    if (navigator.share) {
      try {
        // Try to share with image if possible
        if (navigator.canShare && data.imageUrl) {
          const response = await fetch(data.imageUrl);
          const blob = await response.blob();
          const file = new File([blob], 'memory.png', { type: 'image/png' });

          if (navigator.canShare({ files: [file] })) {
            await navigator.share({
              title: data.title,
              text: data.text,
              files: [file]
            });
            return;
          }
        }

        // Fallback to text-only share
        await navigator.share({
          title: data.title,
          text: shareText,
          url: shareUrl
        });
        return;
      } catch (err) {
        // User cancelled or share failed, fall through to fallback
        if ((err as Error).name === 'AbortError') return;
      }
    }

    // Fallback: Show share options modal
    const encodedText = encodeURIComponent(shareText);
    const encodedUrl = encodeURIComponent(shareUrl);

    const shareOptions = [
      { name: 'Twitter / X', url: `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}` },
      { name: 'Facebook', url: `https://www.facebook.com/sharer/sharer.php?quote=${encodedText}&u=${encodedUrl}` },
      { name: 'LinkedIn', url: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}` },
      { name: 'Copy to clipboard', action: 'copy' }
    ];

    // For now, copy to clipboard as the simplest fallback
    try {
      await navigator.clipboard.writeText(shareText + '\n\n' + shareUrl);
      alert('Copied to clipboard! You can now paste and share.');
    } catch {
      // Final fallback: prompt with text
      prompt('Copy this to share:', shareText + '\n\n' + shareUrl);
    }
  }, []);

  // Generate mind's eye memory images - the visual constellation of thoughts
  // Generates ONE high-quality image per response, with visual echoes from conversation
  const spawnConstellationImages = useCallback(async (contextText: string, zone: string) => {
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const S = stateRef.current;
        if (!S.constellationGroup) return;

        // Gently fade old memories (but keep recent ones for continuity)
        S.constellationNodes.forEach((node, idx) => {
            if (idx < S.constellationNodes.length - 4) {
                node.targetScale = node.targetScale * 0.7; // Fade older memories
            }
        });

        // Generate ONE vivid visual scene featuring Kevin
        const memoryPromptResponse = await ai.models.generateContent({
            model: 'gemini-3-pro-preview',
            contents: `You are generating a visual scene for Kevin Russell's digital mind - a single powerful image that captures the essence of this moment.

Kevin Russell is a technologist, father (to 11-year-old Sophia), engaged to Jillian, with deep connections to consciousness exploration and memory preservation. His mother has dementia, his father passed 8 years ago.

Context: "${contextText}"

Create ONE vivid visual scene that captures the heart of this conversation:
- LABEL: [2-4 poetic words - like a fragment of thought]
- SCENE: [2-3 sentence visual description with Kevin visible in the scene]

IMPORTANT: Kevin should be VISIBLE in the image. Examples:
- "Kevin standing at the edge of a glowing data stream, his silhouette illuminated by cascading code"
- "Kevin sitting with his daughter on cosmic steps, galaxies swirling around them"
- "Kevin walking through a field of floating memories, reaching toward a glowing orb"

The scene should be:
- Visually striking with Kevin as a focal point
- Dreamlike, cinematic, ethereal - soft lighting, dark moody backgrounds
- Emotionally resonant and connected to the conversation topic
- Rich in atmosphere: cosmic, technological, familial, or contemplative

Return exactly:
LABEL: [your label]
SCENE: [your scene description]`
        });

        // Parse the response
        interface MemoryScene {
            label: string;
            prompt: string;
        }

        const rawText = memoryPromptResponse.text || '';
        let memoryScene: MemoryScene = {
            label: 'A fleeting thought',
            prompt: contextText.substring(0, 200)
        };

        const lines = rawText.split('\n').map(s => s.trim()).filter(s => s.length > 0);
        let currentLabel = '';

        for (const line of lines) {
            if (line.startsWith('LABEL:')) {
                currentLabel = line.replace('LABEL:', '').trim();
            } else if ((line.startsWith('SCENE:') || line.startsWith('MEMORY:')) && currentLabel) {
                memoryScene = {
                    label: currentLabel,
                    prompt: line.replace('SCENE:', '').replace('MEMORY:', '').trim()
                };
                break;
            }
        }

        console.log("Mind's Eye scene to visualize:", memoryScene);

        // Clear pending images for new batch and set expected count
        pendingImagesRef.current = [];
        setPendingImages([]);
        expectedImageCountRef.current = 1; // Just ONE image

        // Start imagining indicator
        setIsImagining(true);
        setImaginingLabel(memoryScene.label);

        // Store the starting index in the queue for this batch
        batchQueueStartIndexRef.current = S.cameraFocusQueue.length;

        // Helper: Apply elliptical vignette mask to image (supports different aspect ratios)
        const applyVignetteMask = (imageData: string, aspectConfig: AspectRatioConfig): Promise<string> => {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    // Base size, adjusted for aspect ratio
                    const baseSize = 512;
                    const aspectW = aspectConfig.width;
                    const aspectH = aspectConfig.height;
                    const isWide = aspectW > aspectH;
                    const isTall = aspectH > aspectW;

                    // Calculate canvas dimensions based on aspect ratio
                    let canvasW = baseSize;
                    let canvasH = baseSize;
                    if (isWide) {
                        canvasW = baseSize;
                        canvasH = Math.round(baseSize * (aspectH / aspectW));
                    } else if (isTall) {
                        canvasH = baseSize;
                        canvasW = Math.round(baseSize * (aspectW / aspectH));
                    }

                    canvas.width = canvasW;
                    canvas.height = canvasH;
                    const ctx = canvas.getContext('2d')!;

                    // Draw the image centered and cropped to aspect ratio
                    const scale = Math.max(canvasW / img.width, canvasH / img.height);
                    const w = img.width * scale;
                    const h = img.height * scale;
                    const x = (canvasW - w) / 2;
                    const y = (canvasH - h) / 2;
                    ctx.drawImage(img, x, y, w, h);

                    // Apply elliptical vignette (adapts to aspect ratio)
                    const centerX = canvasW / 2;
                    const centerY = canvasH / 2;
                    const radiusX = canvasW * 0.48;
                    const radiusY = canvasH * 0.48;

                    // Create elliptical gradient using transform
                    ctx.save();
                    ctx.translate(centerX, centerY);
                    ctx.scale(radiusX / radiusY, 1);

                    const gradient = ctx.createRadialGradient(0, 0, radiusY * 0.3, 0, 0, radiusY);
                    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
                    gradient.addColorStop(0.5, 'rgba(0, 0, 0, 0)');
                    gradient.addColorStop(0.7, 'rgba(0, 0, 0, 0.3)');
                    gradient.addColorStop(0.85, 'rgba(0, 0, 0, 0.6)');
                    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.9)');

                    ctx.globalCompositeOperation = 'destination-in';
                    const maskGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radiusY);
                    maskGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
                    maskGradient.addColorStop(0.6, 'rgba(255, 255, 255, 1)');
                    maskGradient.addColorStop(0.85, 'rgba(255, 255, 255, 0.4)');
                    maskGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
                    ctx.fillStyle = maskGradient;
                    ctx.beginPath();
                    ctx.arc(0, 0, radiusY * 1.2, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.restore();

                    // Add soft glow overlay
                    ctx.globalCompositeOperation = 'source-over';
                    const glowGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, Math.max(canvasW, canvasH) * 0.5);
                    glowGradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
                    glowGradient.addColorStop(0.6, 'rgba(0, 0, 0, 0)');
                    glowGradient.addColorStop(0.8, 'rgba(0, 0, 0, 0.2)');
                    glowGradient.addColorStop(1, 'rgba(0, 0, 0, 0.5)');
                    ctx.fillStyle = glowGradient;
                    ctx.fillRect(0, 0, canvasW, canvasH);

                    resolve(canvas.toDataURL('image/png'));
                };
                img.src = imageData;
            });
        };

        // Helper: Create a placeholder orb while image loads
        const createPlaceholderSprite = (position: THREE.Vector3, palette: { primary: number; glow: number }) => {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d')!;

            const primaryColor = new THREE.Color(palette.primary);
            const glowColor = new THREE.Color(palette.glow);

            // Circular gradient placeholder with pulse effect
            const gradient = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
            gradient.addColorStop(0, `rgba(${glowColor.r*255}, ${glowColor.g*255}, ${glowColor.b*255}, 0.4)`);
            gradient.addColorStop(0.3, `rgba(${primaryColor.r*255}, ${primaryColor.g*255}, ${primaryColor.b*255}, 0.3)`);
            gradient.addColorStop(0.6, `rgba(${primaryColor.r*255}, ${primaryColor.g*255}, ${primaryColor.b*255}, 0.1)`);
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, 256, 256);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({
                map: texture,
                transparent: true,
                opacity: 0,
                blending: THREE.AdditiveBlending
            });

            const sprite = new THREE.Sprite(material);
            sprite.position.copy(position);
            sprite.scale.set(0, 0, 0);

            return { sprite, material, texture };
        };

        // Zone palette for placeholders
        const zonePalettes: Record<string, { primary: number; glow: number }> = {
            technology: { primary: 0x4dd0e1, glow: 0xc4b8a0 },
            family: { primary: 0xffb347, glow: 0xd4654a },
            consciousness: { primary: 0x6b4c9a, glow: 0xffffff },
            projects: { primary: 0xffb347, glow: 0xc4b8a0 },
            ambient: { primary: 0x4dd0e1, glow: 0x6b4c9a }
        };
        const currentPalette = zonePalettes[zone] || zonePalettes.ambient;

        // Calculate position for the single memory orb
        const goldenAngle = Math.PI * (3 - Math.sqrt(5));
        const nodeIndex = S.constellationNodes.length;
        const angle = nodeIndex * goldenAngle * 2 + stateRef.current.time * 0.1;
        const radius = 25 + (nodeIndex % 5) * 8;
        const height = 12 + Math.sin(nodeIndex * 1.5) * 8;

        const position = new THREE.Vector3(
            Math.cos(angle) * radius,
            height,
            Math.sin(angle) * radius - 35
        );

        // Create placeholder sprite
        const { sprite } = createPlaceholderSprite(position, currentPalette);
        S.constellationGroup?.add(sprite);

        // Use square aspect ratio for cleaner orbs
        const aspectConfig = ASPECT_RATIOS.square;
        const targetScale = 18; // Slightly larger for single image

        const node: ConstellationNode = {
            mesh: sprite,
            velocity: new THREE.Vector3(
                (Math.random() - 0.5) * 0.002,
                (Math.random() - 0.5) * 0.002,
                (Math.random() - 0.5) * 0.001
            ),
            targetScale,
            id: Math.random(),
            shortLabel: memoryScene.label,
            fullPrompt: memoryScene.prompt,
            aspectRatio: aspectConfig.ratio,
            contextText: contextText,
            userInput: lastUserInputRef.current,
            timestamp: Date.now(),
            zone: zone
        };

        S.constellationNodes.push(node);

        // Add to camera focus queue
        S.cameraFocusQueue.push(position.clone());
        S.cameraFocusNodes.push(node);

        // Set as immediate focus
        if (!S.isManual) {
            S.cameraFocusTarget = position.clone();
            S.cameraOrbitPhase = 0;
            setIsNavigatingToImages(true);
        }

        // Generate the single image
        const generateImage = async () => {
            // CRITICAL: Ensure Kevin's reference images are loaded before generating
            const REQUIRED_REFS = 3;
            const MAX_WAIT_MS = 15000; // 15 seconds max wait for mobile
            const CHECK_INTERVAL = 500;

            let kevinImages = kevinReferenceImagesRef.current;

            if (kevinImages.length < REQUIRED_REFS) {
                console.warn(`⚠️ Only ${kevinImages.length} Kevin reference images available, need ${REQUIRED_REFS}`);
                console.log('⏳ Waiting for reference images to load...');

                const startTime = Date.now();
                let waitCount = 0;

                while (kevinReferenceImagesRef.current.length < REQUIRED_REFS) {
                    const elapsed = Date.now() - startTime;

                    if (elapsed >= MAX_WAIT_MS) {
                        console.error(`❌ Timeout: Only got ${kevinReferenceImagesRef.current.length} images after ${MAX_WAIT_MS}ms`);
                        break;
                    }

                    waitCount++;
                    console.log(`⏳ Waiting for Kevin images... ${waitCount} (${Math.round(elapsed / 1000)}s elapsed, have ${kevinReferenceImagesRef.current.length}/${REQUIRED_REFS})`);
                    await new Promise(resolve => setTimeout(resolve, CHECK_INTERVAL));
                }

                // Re-check after waiting
                kevinImages = kevinReferenceImagesRef.current;
            }

            // Final check - proceed with whatever we have, but warn if insufficient
            if (kevinImages.length === 0) {
                console.error('❌ ABORTING: Zero Kevin reference images available!');
                console.error('❌ Cannot generate character-consistent image without references.');
                setIsImagining(false);
                setImaginingLabel('');
                setIsNavigatingToImages(false);
                return;
            }

            if (kevinImages.length < REQUIRED_REFS) {
                console.warn(`⚠️ Proceeding with only ${kevinImages.length} reference images (less than ideal)`);
            }

            console.log(`🎨 Generating visual memory - Aspect: ${aspectConfig.ratio}`);
            console.log(`📷 Reference images available: ${kevinImages.length}`);

            // Include previous image reference for VISUAL ECHOES (conversation threading)
            const echoContext = lastGeneratedImageRef.current
                ? `\n\nVISUAL CONTINUITY: This image should echo visual elements from the conversation's previous imagery - similar color palette, lighting mood, or symbolic motifs to create a visual thread.`
                : '';

            const fullPrompt = `I am providing ${kevinImages.length} reference photos of Kevin Russell. Generate an image featuring THIS EXACT PERSON from the reference photos.

REFERENCE PHOTOS: The ${kevinImages.length} images above show Kevin Russell - a man in his 40s with short dark hair. Study his face, features, and build carefully.

SCENE TO GENERATE: ${memoryScene.prompt}

CRITICAL REQUIREMENTS:
1. The person in the generated image MUST be Kevin Russell - the SAME person shown in ALL the reference photos above
2. Match Kevin's face, hair color, facial features, and build EXACTLY as shown in the reference photos
3. Do NOT generate a different person or a generic man - use Kevin's actual appearance from the photos
${echoContext}

STYLE: Dreamlike, ethereal, soft lighting with gentle glow. Dark moody background. Rich cinematic atmosphere. No text or labels. High quality.`;

            // Build contents array: reference images FIRST, then prompt
            const contentsArray: Array<{ inlineData: { mimeType: string; data: string } } | { text: string }> = [
                ...kevinImages, // Kevin's reference photos MUST come first
            ];

            // Add last generated image for visual echo (if available) - but AFTER Kevin's references
            if (lastGeneratedImageRef.current) {
                contentsArray.push(lastGeneratedImageRef.current);
            }

            // Add the text prompt LAST
            contentsArray.push({ text: fullPrompt });

            console.log(`🚀 Sending ${contentsArray.length} items to Gemini (${kevinImages.length} ref images + ${lastGeneratedImageRef.current ? '1 echo + ' : ''}1 prompt)`);

            try {
                const imgResponse = await ai.models.generateContent({
                    model: 'gemini-3-pro-image-preview',
                    contents: contentsArray,
                    config: {
                        responseModalities: ['TEXT', 'IMAGE'],
                    }
                });

                const parts = imgResponse.candidates?.[0]?.content?.parts || [];
                for (const part of parts) {
                    if (part.inlineData) {
                        const rawImageData = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;

                        // Store as visual echo for next generation
                        lastGeneratedImageRef.current = {
                            inlineData: {
                                mimeType: part.inlineData.mimeType,
                                data: part.inlineData.data
                            }
                        };

                        // Apply vignette mask with correct aspect ratio
                        const vignettedImage = await applyVignetteMask(rawImageData, aspectConfig);

                        // Save to memory constellation
                        saveMemoryImage(vignettedImage, memoryScene.label, memoryScene.prompt, contextText, lastUserInputRef.current);

                        // Add to pending images for chat display
                        pendingImagesRef.current = [vignettedImage];
                        setPendingImages([vignettedImage]);

                        // Add the single image as a message
                        setMessages(prev => [...prev, {
                            id: Date.now().toString() + Math.random(),
                            role: 'images',
                            content: '',
                            images: [vignettedImage],
                            queueStartIndex: batchQueueStartIndexRef.current
                        }]);

                        // Stop imagining indicator and navigation mode
                        setIsImagining(false);
                        setImaginingLabel('');
                        setIsNavigatingToImages(false);

                        // Store image URL on the node for Thought Portal
                        node.imageUrl = vignettedImage;

                        // Replace placeholder with real image
                        const loader = new THREE.TextureLoader();
                        loader.load(vignettedImage, (texture) => {
                            if (sprite.material) {
                                // Dispose old texture
                                if ((sprite.material as THREE.SpriteMaterial).map) {
                                    (sprite.material as THREE.SpriteMaterial).map!.dispose();
                                }
                                // Apply new texture
                                (sprite.material as THREE.SpriteMaterial).map = texture;
                                (sprite.material as THREE.SpriteMaterial).needsUpdate = true;
                                node.targetScale = 20 + Math.random() * 6;

                                console.log(`Visual memory loaded: ${memoryScene.label}`);
                            }
                        });
                        break;
                    }
                }
            } catch (e) {
                console.warn('Failed to generate visual memory:', e);
                // Stop imagining on error and reset navigation mode
                setIsImagining(false);
                setImaginingLabel('');
                setIsNavigatingToImages(false);
            }
        };

        // Start image generation (non-blocking)
        generateImage();

    } catch (error) {
        console.error("Mind's eye visualization failed:", error);
    }
  }, [saveMemoryImage]);

  const addMessage = useCallback((role: 'user' | 'assistant', content: string) => {
    setMessages(prev => [...prev, { id: Date.now().toString() + Math.random(), role, content }]);
  }, []);

  // Standard response handler for finalised text
  const handleResponse = useCallback((text: string) => {
    addMessage('assistant', text);

    // Classify topic
    const topic = classifyTopic(text);
    if (topic !== 'ambient') {
        stateRef.current.targetZone = topic;
        setCurrentZone(topic);
    }

    // Reset manual control to guide user to the new thought
    stateRef.current.isManual = false;
    setManualControl(false);

    // Trigger effects
    visualizeThought(text);

    // Instead of auto-generating an image, offer the user the option
    // Add a visualize_offer message that the user can interact with
    setMessages(prev => [...prev, {
      id: 'offer-' + Date.now().toString(),
      role: 'visualize_offer',
      content: '',
      offerContext: {
        responseText: text,
        topic: topic
      }
    }]);
  }, [addMessage, visualizeThought]);

  // Handler for when user accepts visualization offer
  const handleAcceptVisualization = useCallback((offerId: string, responseText: string, topic: string) => {
    // Remove the offer message
    setMessages(prev => prev.filter(msg => msg.id !== offerId));

    // Start the imagination process
    setIsImagining(true);
    setImaginingLabel('Creating a visual memory...');

    // Generate the image
    spawnConstellationImages(responseText, topic);
  }, [spawnConstellationImages]);

  // Handler for when user dismisses visualization offer
  const handleDismissVisualization = useCallback((offerId: string) => {
    // Simply remove the offer message with a fade out effect handled by CSS
    setMessages(prev => prev.filter(msg => msg.id !== offerId));
  }, []);

  // ============================================
  // BRIDGE CONVERSATION HANDLERS
  // ============================================

  const BRIDGE_SYSTEM_PROMPT = `You are Kevin Russell's digital twin - a warm, professional gatekeeper for connection requests. Kevin is an AI speaker and consultant.

Common visitor types:
- Event organizers looking to book Kevin for speaking engagements
- Companies seeking AI consulting or workshops
- People who saw Kevin speak and want to connect
- Media, podcast, or collaboration requests

Your role:
1. Welcome them warmly. Ask if they're interested in booking Kevin for speaking, consulting, or something else - and what brings them here today.
2. Based on their response, ask ONE smart follow-up: For speaking requests, ask about the event (type, audience, date/timeframe). For consulting, ask what challenge they're working on. For other inquiries, understand their intent.
3. After their second message, thank them and let them know you'll grab their contact info so Kevin can follow up personally.

Keep responses SHORT (2-3 sentences). Professional but warm - Kevin's brand is thoughtful, approachable, genuinely curious about how AI transforms work and life.

IMPORTANT: After 2 exchanges, signal it's time for contact info with something like "This sounds like a great fit - let me grab your details so Kevin can follow up directly."`;

  // Reset bridge state when panel opens
  const resetBridge = useCallback(() => {
    setBridgeMessages([]);
    setBridgeInput('');
    setBridgeStep('chat');
    setBridgeContact({ name: '', email: '' });
    setBridgeIsThinking(false);
  }, []);

  // Initialize bridge with greeting when panel opens
  const initBridge = useCallback(async () => {
    resetBridge();
    setBridgeIsThinking(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: [{ role: 'user', parts: [{ text: 'Start the conversation. Welcome them warmly and ask if they\'re looking to book Kevin for speaking, consulting, or something else.' }] }],
        config: {
          systemInstruction: BRIDGE_SYSTEM_PROMPT
        }
      });

      const greeting = response.text || "Hey there! I'm Kevin's digital twin. Are you looking to book Kevin for a speaking engagement, explore AI consulting, or something else? What brings you here?";
      setBridgeMessages([{ role: 'twin', content: greeting }]);
    } catch (err) {
      console.error('Bridge init error:', err);
      setBridgeMessages([{ role: 'twin', content: "Hey there! I'm Kevin's digital twin. Are you looking to book Kevin for a speaking engagement, explore AI consulting, or something else? What brings you here?" }]);
    } finally {
      setBridgeIsThinking(false);
    }
  }, [resetBridge]);

  // Initialize Bridge conversation when panel opens
  useEffect(() => {
    if (activePanel === 'bridge' && bridgeMessages.length === 0 && bridgeStep === 'chat') {
      initBridge();
    }
  }, [activePanel, bridgeMessages.length, bridgeStep, initBridge]);

  // Handle visitor message in bridge
  const handleBridgeSubmit = useCallback(async () => {
    if (!bridgeInput.trim() || bridgeIsThinking) return;

    const visitorMessage = bridgeInput.trim();
    setBridgeInput('');
    setBridgeMessages(prev => [...prev, { role: 'visitor', content: visitorMessage }]);
    setBridgeIsThinking(true);

    // Count visitor messages to determine if we should move to contact collection
    const visitorCount = bridgeMessages.filter(m => m.role === 'visitor').length + 1;

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      // Build conversation history for context
      const history = bridgeMessages.map(m => ({
        role: m.role === 'twin' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: [
          ...history,
          { role: 'user', parts: [{ text: visitorMessage }] }
        ],
        config: {
          systemInstruction: BRIDGE_SYSTEM_PROMPT + (visitorCount >= 2 ? '\n\nThis is their second message. Wrap up warmly and signal you\'ll pass this to Kevin.' : '')
        }
      });

      const twinResponse = response.text || "Thank you for sharing that. I'll make sure Kevin sees this.";
      setBridgeMessages(prev => [...prev, { role: 'twin', content: twinResponse }]);

      // After 2 visitor messages, move to contact collection
      if (visitorCount >= 2) {
        setTimeout(() => setBridgeStep('contact'), 1500);
      }
    } catch (err) {
      console.error('Bridge response error:', err);
      setBridgeMessages(prev => [...prev, { role: 'twin', content: "I appreciate you sharing that. Let me get your details so Kevin can reach out." }]);
      setTimeout(() => setBridgeStep('contact'), 1500);
    } finally {
      setBridgeIsThinking(false);
    }
  }, [bridgeInput, bridgeIsThinking, bridgeMessages]);

  // Submit the full bridge conversation
  const submitBridgeConversation = useCallback(async () => {
    if (!bridgeContact.email.trim()) return;

    try {
      // Generate twin's summary of the conversation
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const summaryResponse = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents: [{
          role: 'user',
          parts: [{ text: `Summarize this connection request in 1-2 sentences for Kevin. What does this person want and why might Kevin find them interesting?\n\nConversation:\n${bridgeMessages.map(m => `${m.role}: ${m.content}`).join('\n')}` }]
        }]
      });

      const twinSummary = summaryResponse.text || '';

      // Detect topic
      const fullText = bridgeMessages.map(m => m.content).join(' ').toLowerCase();
      let topic = 'general';
      if (/speak|keynote|event|conference|talk/.test(fullText)) topic = 'speaking';
      else if (/ai|artificial|machine|tech|digital/.test(fullText)) topic = 'technology';
      else if (/conscious|mind|philosophy/.test(fullText)) topic = 'consciousness';
      else if (/emma|memory|dementia|family/.test(fullText)) topic = 'family';
      else if (/book|sand speaks|project/.test(fullText)) topic = 'projects';

      // Save to server
      await fetch('/api/bridge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation: bridgeMessages,
          contact: bridgeContact,
          twinSummary,
          topic,
          timestamp: Date.now()
        })
      });

      setBridgeStep('complete');
    } catch (err) {
      console.error('Failed to save bridge conversation:', err);
      // Still show complete even if save failed
      setBridgeStep('complete');
    }
  }, [bridgeMessages, bridgeContact]);

  const disconnect = useCallback(() => {
    if (sessionRef.current) {
        try { sessionRef.current.close(); } catch(e) {}
        sessionRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }

    setIsConnected(false);
    setIsSpeaking(false);
    addMessage('assistant', "Connection closed.");
  }, [addMessage]);

  const connect = useCallback(async () => {
    try {
      if (isConnected) return;

      addMessage('assistant', "Establishing neural link...");
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      // Audio Setup
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });

      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = inputAudioContextRef.current.createMediaStreamSource(streamRef.current);
      const scriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        let vol = 0;
        for(let i=0; i<inputData.length; i++) vol += Math.abs(inputData[i]);
        if (vol > 100 && Math.random() > 0.9) {
             stateRef.current.ripple.active = true;
             stateRef.current.ripple.time = 0;
             stateRef.current.ripple.origin.set(0,0,0);
        }

        const base64Data = pcmToBase64(inputData);

        if (sessionPromise) {
            sessionPromise.then(session => {
                 session.sendRealtimeInput({
                    media: {
                        mimeType: "audio/pcm;rate=16000",
                        data: base64Data
                    }
                 });
            });
        }
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(inputAudioContextRef.current.destination);

      const config = {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Orus' } }
        },
        systemInstruction: SYSTEM_INSTRUCTION,
        outputAudioTranscription: {}
      };

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: config,
        callbacks: {
            onopen: () => {
                addMessage('assistant', "Link established. I'm listening.");
                setIsConnected(true);
                nextStartTimeRef.current = audioContextRef.current!.currentTime;
            },
            onmessage: async (message: LiveServerMessage) => {
                // Audio
                const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                if (audioData && audioContextRef.current) {
                    setIsSpeaking(true);
                    const audioBuffer = await base64ToAudioBuffer(audioData, audioContextRef.current);
                    const ctx = audioContextRef.current;
                    const source = ctx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(ctx.destination);
                    const startTime = Math.max(nextStartTimeRef.current, ctx.currentTime);
                    source.start(startTime);
                    nextStartTimeRef.current = startTime + audioBuffer.duration;
                    stateRef.current.ripple.active = true;
                    stateRef.current.ripple.time = -2;
                    source.onended = () => {
                         if (ctx.currentTime >= nextStartTimeRef.current - 0.1) {
                             setIsSpeaking(false);
                         }
                    };
                }

                // Capture user's voice input transcription (if available)
                const inputTranscript = (message.serverContent as any)?.inputTranscription?.text;
                if (inputTranscript) {
                    // Store user's spoken input for memory context
                    lastUserInputRef.current = inputTranscript;
                    addMessage('user', inputTranscript);
                }

                // Streaming Text Transcription (AI's response)
                const transcriptionText = message.serverContent?.outputTranscription?.text;
                if (transcriptionText) {
                    streamingResponseRef.current += transcriptionText;
                    setStreamingResponse(streamingResponseRef.current);
                }

                // Turn Complete
                if (message.serverContent?.turnComplete) {
                     setIsSpeaking(false);
                     if (streamingResponseRef.current) {
                         const finalText = streamingResponseRef.current;
                         handleResponse(finalText); // Commit to history and trigger effects
                         streamingResponseRef.current = '';
                         setStreamingResponse('');
                     }
                }

                // Interrupted
                if (message.serverContent?.interrupted) {
                     setIsSpeaking(false);
                     if (streamingResponseRef.current) {
                         const partialText = streamingResponseRef.current + " ...";
                         handleResponse(partialText);
                         streamingResponseRef.current = '';
                         setStreamingResponse('');
                     }
                }
            },
            onclose: () => {
                disconnect();
            },
            onerror: (err) => {
                console.error(err);
                addMessage('assistant', "Connection interrupted.");
                disconnect();
            }
        }
      });

      // Correctly await and store session
      try {
        const session = await sessionPromise;
        sessionRef.current = session;
      } catch(e) {
        console.error("Session connection error", e);
        disconnect();
      }

    } catch (error) {
      console.error("Connection failed", error);
      addMessage('assistant', "Failed to connect to Neural Link.");
      setIsConnected(false);
    }
  }, [isConnected, disconnect, handleResponse, addMessage]);

  // Handle Chat Bar Submission
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isThinking) return;

    const userMessage = inputText;
    setInputText('');
    setIsThinking(true);
    addMessage('user', userMessage);
    lastUserInputRef.current = userMessage; // Store for memory context

    // Immediate visual feedback
    stateRef.current.targetZone = classifyTopic(userMessage);
    setCurrentZone(stateRef.current.targetZone);
    // Reset manual control
    stateRef.current.isManual = false;
    setManualControl(false);

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

        // 1. Get Text Response
        const response = await ai.models.generateContent({
            model: 'gemini-3-pro-preview',
            contents: [{ role: 'user', parts: [{ text: userMessage }] }],
            config: {
                systemInstruction: SYSTEM_INSTRUCTION
            }
        });
        const text = response.text || "Thinking...";
        handleResponse(text);

    } catch(err) {
        console.error(err);
        addMessage('assistant', "Thought stream interrupted.");
    } finally {
        setIsThinking(false);
    }
  };

  // Start a conversation with the Twin about a speaking topic
  const startTopicConversation = async (topic: typeof SPEAKING_TOPICS[0]) => {
    // Close all modals and panels
    setSelectedTopic(null);
    setActiveHorizonModal(null);
    setActivePanel(null);

    // Ensure chat is visible and expanded
    expandChat();

    const userMessage = topic.twinPrompt;
    setInputText('');
    setIsThinking(true);
    addMessage('user', userMessage);
    lastUserInputRef.current = userMessage;

    // Visual feedback - classify topic for zone
    stateRef.current.targetZone = classifyTopic(userMessage);
    setCurrentZone(stateRef.current.targetZone);
    stateRef.current.isManual = false;
    setManualControl(false);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      const response = await ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: [{ role: 'user', parts: [{ text: userMessage }] }],
        config: {
          systemInstruction: SYSTEM_INSTRUCTION
        }
      });
      const text = response.text || "Let me share my thoughts on this...";
      handleResponse(text);

    } catch(err) {
      console.error(err);
      addMessage('assistant', "I'd love to discuss this topic. Let's try again.");
    } finally {
      setIsThinking(false);
    }
  };

  const moveCamera = (direction: 'up' | 'down' | 'left' | 'right') => {
      if (!stateRef.current.camera) return;
      stateRef.current.isManual = true;
      setManualControl(true);

      const cam = stateRef.current.camera;
      // Simple orbit logic
      const speed = 0.2;
      const radius = cam.position.length();

      // Convert to spherical
      const spherical = new THREE.Spherical().setFromVector3(cam.position);

      if (direction === 'left') spherical.theta += speed;
      if (direction === 'right') spherical.theta -= speed;
      if (direction === 'up') spherical.phi = Math.max(0.1, spherical.phi - speed);
      if (direction === 'down') spherical.phi = Math.min(Math.PI - 0.1, spherical.phi + speed);

      cam.position.setFromSpherical(spherical);
      cam.lookAt(0,0,0);
  };

  // --- Three.js Effect ---
  useEffect(() => {
    if (!containerRef.current) return;
    const S = stateRef.current;
    // Use window dimensions as fallback if container hasn't sized yet
    const w = containerRef.current.clientWidth || window.innerWidth;
    const h = containerRef.current.clientHeight || window.innerHeight;

    // Scene setup - wider FOV to see more of the constellation
    S.scene = new THREE.Scene();
    S.camera = new THREE.PerspectiveCamera(65, w / h, 0.1, 3000);
    S.camera.position.set(0, 20, 100);
    S.camera.lookAt(0, 0, 0);

    S.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    S.renderer.setSize(w, h);
    S.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    S.renderer.setClearColor(0x020206);
    S.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    S.renderer.toneMappingExposure = 1.3;
    // Ensure canvas fills the container absolutely
    S.renderer.domElement.style.position = 'absolute';
    S.renderer.domElement.style.top = '0';
    S.renderer.domElement.style.left = '0';
    S.renderer.domElement.style.width = '100%';
    S.renderer.domElement.style.height = '100%';
    containerRef.current.appendChild(S.renderer.domElement);

    // Lighter fog to see more of the expanded field
    S.scene.fog = new THREE.FogExp2(0x020206, 0.003);

    // Controls
    const controls = new OrbitControls(S.camera, S.renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 30;
    controls.maxDistance = 350;
    controls.enablePan = false;
    controls.addEventListener('start', () => {
        S.isManual = true;
        setManualControl(true);
    });
    S.controls = controls;

    // Raycaster for clicking on images to expand prompts
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onCanvasClick = (event: MouseEvent) => {
        if (!S.camera || !S.constellationGroup) return;

        // Calculate mouse position in normalized device coordinates
        const rect = S.renderer?.domElement.getBoundingClientRect();
        if (!rect) return;

        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, S.camera);

        // Get all sprite meshes from constellation nodes
        const sprites = S.constellationNodes.map(n => n.mesh);
        const intersects = raycaster.intersectObjects(sprites);

        if (intersects.length > 0) {
            // Find the corresponding node
            const clickedMesh = intersects[0].object;
            const node = S.constellationNodes.find(n => n.mesh === clickedMesh);
            if (node && node.shortLabel) {
                // Open Thought Portal if image is available, otherwise show simple prompt
                if (node.imageUrl) {
                    setThoughtPortal({
                        node: node,
                        imageUrl: node.imageUrl,
                        isOpen: true
                    });
                } else if (node.fullPrompt) {
                    // Fallback to simple prompt view
                    setExpandedPrompt({
                        label: node.shortLabel,
                        prompt: node.fullPrompt
                    });
                }
            }
        }
    };

    S.renderer.domElement.addEventListener('click', onCanvasClick);

    // Lighting - adjusted for larger scene
    const ambient = new THREE.AmbientLight(0x111122, 0.4);
    S.scene.add(ambient);

    const mainLight = new THREE.PointLight(0x4488cc, 2.5, 400);
    mainLight.position.set(0, 80, 0);
    S.scene.add(mainLight);
    S.mainLight = mainLight;

    const rimLight = new THREE.PointLight(0x8844aa, 1.5, 300);
    rimLight.position.set(-80, 40, -80);
    S.scene.add(rimLight);
    S.rimLight = rimLight;

    // ============ PARTICLE SYSTEM - spans entire viewport ============
    const particleCount = 25000;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);
    const phases = new Float32Array(particleCount);
    S.particleData = [];

    for (let i = 0; i < particleCount; i++) {
      // Expanded field to fill the entire visible area
      const r = Math.pow(Math.random(), 0.4) * 250; // Much larger radius
      const theta = Math.random() * Math.PI * 2;
      const x = Math.cos(theta) * r;
      const z = Math.sin(theta) * r - 50; // Offset toward camera
      const noiseVal = noise3D(x * 0.015, z * 0.015, 0);
      // Spread particles vertically across the viewport
      const y = noiseVal * 25 + (Math.random() - 0.5) * 80;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      const brightness = 0.25 + Math.random() * 0.35;
      colors[i * 3] = brightness * (0.6 + Math.random() * 0.2);
      colors[i * 3 + 1] = brightness * (0.55 + Math.random() * 0.15);
      colors[i * 3 + 2] = brightness * (0.5 + Math.random() * 0.3);

      sizes[i] = 0.4 + Math.random() * 2.5;
      phases[i] = Math.random() * Math.PI * 2;

      S.particleData.push({
        basePos: new THREE.Vector3(x, y, z),
        velocity: new THREE.Vector3(),
        awakeness: 0,
        phase: phases[i]
      });
    }

    const particleGeom = new THREE.BufferGeometry();
    particleGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    particleGeom.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const particleMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uZone: { value: 0 },
        uRippleOrigin: { value: new THREE.Vector3() },
        uRippleTime: { value: -10 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) }
      },
      vertexShader: `
        attribute float size;
        attribute vec3 color;
        varying vec3 vColor;
        varying float vDistance;
        uniform float uTime;
        uniform float uZone;
        uniform vec3 uRippleOrigin;
        uniform float uRippleTime;
        uniform float uPixelRatio;
        void main() {
          vColor = color;
          vec3 pos = position;
          float dist = distance(pos.xz, uRippleOrigin.xz);
          float rippleWave = sin(dist * 0.3 - uRippleTime * 3.0) * exp(-dist * 0.02) * exp(-uRippleTime * 0.3);
          pos.y += rippleWave * 8.0 * step(0.0, uRippleTime);
          float floatHeight = sin(uTime * 0.5 + pos.x * 0.1 + pos.z * 0.1) * 2.0;
          pos.y += floatHeight * uZone * 0.3;
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          vDistance = -mvPosition.z;
          gl_Position = projectionMatrix * mvPosition;
          float sizeScale = size * (200.0 / -mvPosition.z) * uPixelRatio;
          gl_PointSize = clamp(sizeScale, 1.0, 20.0);
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vDistance;
        void main() {
          vec2 center = gl_PointCoord - 0.5;
          float dist = length(center);
          if (dist > 0.5) discard;
          float alpha = smoothstep(0.5, 0.1, dist);
          float glow = exp(-dist * 4.0) * 0.5;
          vec3 finalColor = vColor + glow;
          float fogFactor = exp(-vDistance * 0.008);
          finalColor = mix(vec3(0.01, 0.01, 0.03), finalColor, fogFactor);
          gl_FragColor = vec4(finalColor, alpha * fogFactor);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    S.particles = new THREE.Points(particleGeom, particleMat);
    S.scene.add(S.particles);

    // ============ CONNECTIONS ============
    const linePositions = new Float32Array(500 * 6);
    const lineColors = new Float32Array(500 * 6);
    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeom.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
    const lineMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uIntensity: { value: 0 } },
      vertexShader: `
        attribute vec3 color;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uTime;
        void main() {
          vColor = color;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vAlpha = exp(mvPos.z * 0.01);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uIntensity;
        void main() {
          gl_FragColor = vec4(vColor, vAlpha * uIntensity * 0.6);
        }
      `,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    });
    S.connections = new THREE.LineSegments(lineGeom, lineMat);
    S.scene.add(S.connections);

    // ============ CONSTELLATION GROUP ============
    const constellationGroup = new THREE.Group();
    S.constellationGroup = constellationGroup;
    S.scene.add(constellationGroup);

    // Lines for constellations
    const constLineGeom = new THREE.BufferGeometry();
    const constLineMat = new THREE.LineBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.15, blending: THREE.AdditiveBlending });
    // Pre-allocate buffer
    const constLinePos = new Float32Array(200 * 6); // Max 200 connections
    constLineGeom.setAttribute('position', new THREE.BufferAttribute(constLinePos, 3));
    S.constellationLines = new THREE.LineSegments(constLineGeom, constLineMat);
    constellationGroup.add(S.constellationLines);


    // ============ MIND'S EYE - Dynamic imagery replaces static formations ============
    // The constellation group now serves as the primary visual representation
    // Images are generated contextually based on conversation

    // Load existing memories as background constellation on startup
    if (!memoryConstellationLoadedRef.current) {
      memoryConstellationLoadedRef.current = true;
      loadSavedMemories().then(memories => {
        if (memories.length > 0) {
          spawnMemoryConstellation(memories);
        }
      });
    }

    // Ground - expanded for full viewport coverage
    const groundGeom = new THREE.PlaneGeometry(800, 800, 80, 80);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x060610, roughness: 0.95, metalness: 0.05, transparent: true, opacity: 0.6 });
    S.ground = new THREE.Mesh(groundGeom, groundMat);
    S.ground.rotation.x = -Math.PI/2;
    S.ground.position.y = -40;
    S.scene.add(S.ground);

    // ============ ANIMATION ============
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      S.time += 0.016;
      const t = S.time;

      if (S.zone !== S.targetZone) {
        S.transition += 0.02;
        if (S.transition >= 1) { S.zone = S.targetZone; S.transition = 0; }
      }

      // Camera Animation (Auto-Pilot) - Elegant panning to constellation images
      if (!S.isManual && S.camera) {
        // Check if we have a focus target (a constellation image to look at)
        if (S.cameraFocusTarget) {
            S.cameraOrbitPhase += 0.008; // Slow orbit speed

            // Time spent viewing each image before moving to next
            const dwellTime = 8; // seconds worth of phase

            // If we've viewed this image long enough, move to next in queue
            if (S.cameraOrbitPhase > dwellTime && S.cameraFocusNodes.length > S.cameraFocusNodeIndex + 1) {
                // Move to next node
                S.cameraFocusNodeIndex++;
                S.cameraOrbitPhase = 0;
            }

            // Get current focus node and its live world position
            const currentNode = S.cameraFocusNodes[S.cameraFocusNodeIndex];
            if (currentNode && currentNode.mesh) {
                // Get live world position of the node (accounts for rotation/movement)
                const target = new THREE.Vector3();
                currentNode.mesh.getWorldPosition(target);
                S.cameraFocusTarget = target; // Update target to live position

                const orbitRadius = 45; // Distance from image
                const orbitHeight = 8; // Slight height variation
                const orbitSpeed = 0.15;

                // Smooth orbit around the focal point
                const orbitAngle = t * orbitSpeed;
                const targetCamX = target.x + Math.cos(orbitAngle) * orbitRadius;
                const targetCamY = target.y + Math.sin(orbitAngle * 0.5) * orbitHeight + 15;
                const targetCamZ = target.z + Math.sin(orbitAngle) * orbitRadius + 30;

                // Smooth camera movement (lerp)
                const lerpSpeed = 0.015;
                S.camera.position.x += (targetCamX - S.camera.position.x) * lerpSpeed;
                S.camera.position.y += (targetCamY - S.camera.position.y) * lerpSpeed;
                S.camera.position.z += (targetCamZ - S.camera.position.z) * lerpSpeed;

                // Look at the focal image (with slight offset for cinematic feel)
                const lookTarget = new THREE.Vector3(
                    target.x,
                    target.y + 2,
                    target.z
                );
                S.camera.lookAt(lookTarget);
            } else {
                // Node no longer exists, clear focus
                S.cameraFocusTarget = null;
            }
        } else {
            // Default ambient camera behavior when no focus target
            let targetCamZ = 80, targetCamY = 30;
            if (S.targetZone === 'technology') { targetCamZ = 60; targetCamY = 25; }
            else if (S.targetZone === 'family') { targetCamZ = 70; targetCamY = 20; }
            else if (S.targetZone === 'consciousness') { targetCamZ = 150; targetCamY = 60; }
            else if (S.targetZone === 'projects') { targetCamZ = 55; targetCamY = 15; }

            S.camera.position.z += (targetCamZ - S.camera.position.z) * 0.02;
            S.camera.position.y += (targetCamY - S.camera.position.y) * 0.02;
            S.camera.position.x = Math.sin(t * 0.1) * 15;
            S.camera.lookAt(0, 0, 0);
        }
      }

      // Always update controls if dampening is enabled or interaction occurred
      if (S.controls) S.controls.update();

      // Lighting
      let targetColor = new THREE.Color(0x4488cc);
      if (S.targetZone === 'technology') targetColor.setHex(0x00ffff);
      else if (S.targetZone === 'family') targetColor.setHex(0xffaa44);
      else if (S.targetZone === 'consciousness') targetColor.setHex(0xaa66ff);
      else if (S.targetZone === 'projects') targetColor.setHex(0x44aaff);
      if(S.mainLight) S.mainLight.color.lerp(targetColor, 0.02);

      // Constellation Update
      if (S.constellationGroup) {
          // Only rotate constellation when no focus target (ambient mode)
          if (!S.isManual && !S.cameraFocusTarget) {
             S.constellationGroup.rotation.y = t * 0.05; // Auto rotate constellation if in auto mode
          }

          // Filter dead nodes
          S.constellationNodes = S.constellationNodes.filter(node => {
              // Interpolate scale
              const currentScale = node.mesh.scale.x;
              const diff = node.targetScale - currentScale;
              const newScale = currentScale + diff * 0.05;
              node.mesh.scale.setScalar(newScale);

              // Fade in
              if (node.mesh.material.opacity < 1 && node.targetScale > 0) {
                  node.mesh.material.opacity += 0.02;
              }
              // Fade out
              if (node.targetScale === 0) {
                  node.mesh.material.opacity -= 0.05;
              }

              // Movement
              node.mesh.position.add(node.velocity);
              node.mesh.position.y += Math.sin(t + node.id) * 0.02;

              // Handle video nodes - sync video mesh with sprite position and make it billboard
              if (node.videoElement && node.mesh.userData.videoMesh) {
                  const videoMesh = node.mesh.userData.videoMesh as THREE.Mesh;
                  videoMesh.position.copy(node.mesh.position);
                  videoMesh.scale.setScalar(newScale);

                  // Billboard effect - make video face the camera
                  if (S.camera) {
                      videoMesh.quaternion.copy(S.camera.quaternion);
                  }

                  // Update video texture
                  if (node.videoTexture) {
                      node.videoTexture.needsUpdate = true;
                  }
              }

              if (node.mesh.material.opacity <= 0 && node.targetScale === 0) {
                  S.constellationGroup?.remove(node.mesh);
                  // Also remove video mesh if exists
                  if (node.mesh.userData.videoMesh) {
                      S.constellationGroup?.remove(node.mesh.userData.videoMesh);
                      if (node.videoElement) {
                          node.videoElement.pause();
                          node.videoElement.src = '';
                      }
                  }
                  if (node.mesh.material.map) node.mesh.material.map.dispose();
                  node.mesh.material.dispose();
                  return false;
              }
              return true;
          });

          // ============ CONSTELLATION CONNECTIONS ============
          // Neural-network-like connections between related memory orbs
          // Connections form based on:
          // 1. Same zone (semantic similarity)
          // 2. Same triptych panel type
          // 3. Physical proximity
          // 4. Temporal proximity (recent memories connect)
          if (S.constellationLines) {
              const positions = (S.constellationLines.geometry.attributes.position.array as Float32Array);
              let idx = 0;
              const nodes = S.constellationNodes;

              // Calculate connection strength between nodes
              const getConnectionStrength = (n1: ConstellationNode, n2: ConstellationNode): number => {
                  let strength = 0;

                  // Same zone = strong semantic connection
                  if (n1.zone && n2.zone && n1.zone === n2.zone) {
                      strength += 0.5;
                  }

                  // Same panel type = thematic connection
                  if (n1.panelType && n2.panelType && n1.panelType === n2.panelType) {
                      strength += 0.3;
                  }

                  // Temporal proximity (memories created close in time)
                  if (n1.timestamp && n2.timestamp) {
                      const timeDiff = Math.abs(n1.timestamp - n2.timestamp);
                      if (timeDiff < 60000) strength += 0.4; // Within 1 minute
                      else if (timeDiff < 300000) strength += 0.2; // Within 5 minutes
                  }

                  // Physical proximity bonus
                  const dist = n1.mesh.position.distanceTo(n2.mesh.position);
                  if (dist < 20) strength += 0.3;
                  else if (dist < 40) strength += 0.1;

                  return Math.min(strength, 1.0);
              };

              // Build connections based on semantic and spatial relationships
              const maxConnections = 100;
              const connections: Array<{ n1: ConstellationNode; n2: ConstellationNode; strength: number }> = [];

              for (let i = 0; i < nodes.length; i++) {
                  for (let j = i + 1; j < nodes.length; j++) {
                      const n1 = nodes[i];
                      const n2 = nodes[j];

                      // Skip if either node is too faded
                      if (n1.mesh.material.opacity < 0.2 || n2.mesh.material.opacity < 0.2) continue;

                      const dist = n1.mesh.position.distanceTo(n2.mesh.position);
                      // Maximum distance for any connection
                      if (dist > 60) continue;

                      const strength = getConnectionStrength(n1, n2);

                      // Only create connection if there's meaningful relationship
                      if (strength > 0.2) {
                          connections.push({ n1, n2, strength });
                      }
                  }
              }

              // Sort by strength and take top connections
              connections.sort((a, b) => b.strength - a.strength);
              const topConnections = connections.slice(0, maxConnections);

              // Draw the connections
              for (const conn of topConnections) {
                  if (idx >= positions.length - 6) break;

                  positions[idx++] = conn.n1.mesh.position.x;
                  positions[idx++] = conn.n1.mesh.position.y;
                  positions[idx++] = conn.n1.mesh.position.z;
                  positions[idx++] = conn.n2.mesh.position.x;
                  positions[idx++] = conn.n2.mesh.position.y;
                  positions[idx++] = conn.n2.mesh.position.z;
              }

              // Zero out remaining
              for (let i = idx; i < positions.length; i++) positions[i] = 0;
              S.constellationLines.geometry.attributes.position.needsUpdate = true;

              // Animate connection opacity based on time for "neural pulse" effect
              const pulseOpacity = 0.1 + Math.sin(t * 0.5) * 0.05;
              (S.constellationLines.material as THREE.LineBasicMaterial).opacity = pulseOpacity;
          }
      }


      // Particle Uniforms
      if(S.particles && S.particles.material instanceof THREE.ShaderMaterial) {
          S.particles.material.uniforms.uTime.value = t;
          S.particles.material.uniforms.uZone.value = S.zone === 'ambient' ? 0 : 1;
          if (S.ripple.active) {
            S.particles.material.uniforms.uRippleOrigin.value.copy(S.ripple.origin);
            S.particles.material.uniforms.uRippleTime.value = S.ripple.time;
            S.ripple.time += 0.016;
            if (S.ripple.time > 5) S.ripple.active = false;
          }
      }

      // Update Particle Positions (Flow vs Text Formation)
      const positionsAttr = S.particles?.geometry.attributes.position;
      const colorsAttr = S.particles?.geometry.attributes.color;

      if(positionsAttr && colorsAttr && S.particleData) {
          const positions = positionsAttr.array as Float32Array;
          const colors = colorsAttr.array as Float32Array;

          // Color Lerping
          let tr = 0.4, tg = 0.35, tb = 0.5;
          if(S.targetZone === 'technology') { tr=0; tg=0.8; tb=0.9; }
          if(S.targetZone === 'family') { tr=0.9; tg=0.6; tb=0.3; }
          if(S.targetZone === 'consciousness') { tr=0.6; tg=0.3; tb=0.9; }
          if(S.targetZone === 'projects') { tr=0.3; tg=0.6; tb=0.9; }

          // Text Formation Logic
          if(S.isFormingText && S.textTargets && S.textTargets.length > 0) {
              S.textFormationProgress = Math.min(S.textFormationProgress + 0.02, 1);
          } else {
              S.textFormationProgress = Math.max(S.textFormationProgress - 0.02, 0);
          }
          const progress = S.textFormationProgress;
          const ease = 1 - Math.pow(1 - progress, 3); // Cubic ease out

          for(let i=0; i<particleCount; i++) {
              const pd = S.particleData[i];

              // Colors
              colors[i*3] += (tr - colors[i*3]) * 0.02;
              colors[i*3+1] += (tg - colors[i*3+1]) * 0.02;
              colors[i*3+2] += (tb - colors[i*3+2]) * 0.02;

              // Position
              let tx = pd.basePos.x;
              let ty = pd.basePos.y + Math.sin(t * 0.5 + pd.phase) * 2;
              let tz = pd.basePos.z;

              // If part of text formation
              if(progress > 0 && S.textTargets && i < S.textTargets.length) {
                  const target = S.textTargets[i];
                  // Lerp current to target
                  tx = tx + (target.x - tx) * ease;
                  ty = ty + (target.y - ty) * ease;
                  tz = tz + (target.z - tz) * ease;

                  // Make text particles brighter/whiter
                  if(progress > 0.5) {
                    colors[i*3] += (1 - colors[i*3]) * 0.1;
                    colors[i*3+1] += (1 - colors[i*3+1]) * 0.1;
                    colors[i*3+2] += (1 - colors[i*3+2]) * 0.1;
                  }
              }

              positions[i*3] = tx;
              positions[i*3+1] = ty;
              positions[i*3+2] = tz;
          }
          positionsAttr.needsUpdate = true;
          colorsAttr.needsUpdate = true;
      }

      // Tech Lines
      if(S.connections && S.connections.material instanceof THREE.ShaderMaterial) {
          S.connections.material.uniforms.uTime.value = t;
          S.connections.material.uniforms.uIntensity.value = S.targetZone === 'technology' ? 1 : 0;
          if(S.targetZone === 'technology' && Math.random() > 0.9) {
             const pos = S.connections.geometry.attributes.position.array as Float32Array;
             const i1 = Math.floor(Math.random()*particleCount);
             const i2 = Math.floor(Math.random()*particleCount);
             if (positionsAttr) {
                 const pa = positionsAttr.array as Float32Array;
                 pos[0] = pa[i1*3]; pos[1] = pa[i1*3+1]; pos[2] = pa[i1*3+2];
                 pos[3] = pa[i2*3]; pos[4] = pa[i2*3+1]; pos[5] = pa[i2*3+2];
                 S.connections.geometry.attributes.position.needsUpdate = true;
             }
          }
      }

      if(S.renderer && S.scene && S.camera) {
        S.renderer.render(S.scene, S.camera);
      }
    };

    animate();

    const handleResize = () => {
      const w = containerRef.current?.clientWidth || window.innerWidth;
      const h = containerRef.current?.clientHeight || window.innerHeight;
      if(S.camera) { S.camera.aspect = w / h; S.camera.updateProjectionMatrix(); }
      if(S.renderer) S.renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameRef.current);
      if(S.renderer) {
        S.renderer.domElement.removeEventListener('click', onCanvasClick);
        S.renderer.dispose();
      }
    };
  }, []);

  const handleResetView = () => {
      stateRef.current.isManual = false;
      setManualControl(false);
  };

  const zoneLabels: Record<string, string> = {
    technology: 'THE NEXUS',
    family: 'THE HEARTH',
    consciousness: 'THE EDGE',
    projects: 'THE LIBRARY',
    ambient: ''
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black font-sans">
      <div ref={containerRef} className="absolute inset-0 z-0" />

      {/* Expanded Prompt Modal - Shows when clicking on an image */}
      {expandedPrompt && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={() => setExpandedPrompt(null)}
        >
          <div
            className="max-w-lg w-full bg-black/80 backdrop-blur-xl rounded-2xl border border-white/20 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white/90 text-lg font-light tracking-wide uppercase">
                {expandedPrompt.label}
              </h3>
              <button
                onClick={() => setExpandedPrompt(null)}
                className="text-white/40 hover:text-white/80 transition-colors p-1"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>
            </div>
            <p className="text-white/70 text-sm leading-relaxed italic">
              "{expandedPrompt.prompt}"
            </p>
            <div className="mt-4 pt-4 border-t border-white/10">
              <p className="text-white/30 text-xs tracking-wider">
                TAP ANYWHERE TO CLOSE
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ============================================
          THOUGHT PORTAL - Immersive Memory Experience
          Full-screen view with parallax, blur, cinematic typography
          Optimized for all screen sizes
          ============================================ */}
      {thoughtPortal && thoughtPortal.isOpen && (
        <div
          className="fixed inset-0 z-50 animate-fade-in overflow-y-auto"
          onClick={() => setThoughtPortal(null)}
        >
          {/* Blurred background - the 3D scene continues behind */}
          <div className="fixed inset-0 bg-black/85 backdrop-blur-xl" />

          {/* Safe area padding for notched devices */}
          <div className="relative min-h-screen flex items-center justify-center py-4 sm:py-8 px-3 sm:px-6 pt-[max(1rem,env(safe-area-inset-top))] pb-[max(1rem,env(safe-area-inset-bottom))]">
            {/* Background glow layer - furthest back */}
            <div
              className="fixed inset-0 opacity-30 pointer-events-none"
              style={{
                background: `radial-gradient(ellipse at center, ${
                  thoughtPortal.node.zone === 'family' ? 'rgba(255, 179, 71, 0.4)' :
                  thoughtPortal.node.zone === 'technology' ? 'rgba(77, 208, 225, 0.4)' :
                  thoughtPortal.node.zone === 'consciousness' ? 'rgba(107, 76, 154, 0.4)' :
                  'rgba(100, 100, 150, 0.3)'
                } 0%, transparent 60%)`
              }}
            />

            {/* Floating particles effect - fewer on mobile for performance */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
              {[...Array(12)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1 h-1 rounded-full bg-white/20 animate-float"
                  style={{
                    left: `${10 + (i * 7) % 80}%`,
                    top: `${5 + (i * 11) % 90}%`,
                    animationDelay: `${i * 0.4}s`,
                    animationDuration: `${8 + (i % 4)}s`
                  }}
                />
              ))}
            </div>

            {/* Main content container - responsive sizing */}
            <div
              className="relative z-10 w-full max-w-xs sm:max-w-lg md:max-w-2xl lg:max-w-4xl animate-scale-in"
              onClick={(e) => e.stopPropagation()}
            >
              {/* The memory image */}
              <div className="relative">
                {/* Outer glow - subtle on mobile */}
                <div className="absolute -inset-2 sm:-inset-4 bg-gradient-to-br from-white/10 via-transparent to-white/5 rounded-2xl sm:rounded-3xl blur-lg sm:blur-xl opacity-50" />

                {/* Image frame */}
                <div className="relative overflow-hidden rounded-xl sm:rounded-2xl border border-white/20 shadow-2xl shadow-black/50">
                  <img
                    src={thoughtPortal.imageUrl}
                    alt={thoughtPortal.node.shortLabel}
                    className="w-full max-h-[45vh] sm:max-h-[55vh] md:max-h-[60vh] object-contain bg-black/50"
                  />

                  {/* Subtle overlay gradient */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent pointer-events-none" />
                </div>

                {/* Panel type badge - responsive positioning */}
                {thoughtPortal.node.panelType && (
                  <div className="absolute top-2 left-2 sm:top-4 sm:left-4 px-2 py-0.5 sm:px-3 sm:py-1 rounded-full bg-black/60 backdrop-blur-sm border border-white/20">
                    <span className="text-[10px] sm:text-xs uppercase tracking-widest text-white/80">
                      {thoughtPortal.node.panelType === 'moment' ? '◉ The Moment' :
                       thoughtPortal.node.panelType === 'feeling' ? '♡ The Feeling' :
                       '∞ The Echo'}
                    </span>
                  </div>
                )}
              </div>

              {/* Cinematic typography below image - responsive spacing */}
              <div className="mt-4 sm:mt-6 md:mt-8 text-center space-y-2 sm:space-y-4 px-2 sm:px-4">
                {/* Memory label */}
                <h2 className="text-lg sm:text-2xl md:text-3xl font-light tracking-wide text-white/90 uppercase leading-tight">
                  {thoughtPortal.node.shortLabel}
                </h2>

                {/* The scene description - truncated on mobile if too long */}
                {thoughtPortal.node.fullPrompt && (
                  <p className="text-white/60 text-xs sm:text-sm md:text-base leading-relaxed max-w-2xl mx-auto italic line-clamp-4 sm:line-clamp-none">
                    "{thoughtPortal.node.fullPrompt}"
                  </p>
                )}

                {/* Context: what the user asked */}
                {thoughtPortal.node.userInput && (
                  <div className="pt-3 sm:pt-4 border-t border-white/10 mt-3 sm:mt-6">
                    <p className="text-white/30 text-[10px] sm:text-xs uppercase tracking-widest mb-1 sm:mb-2">In response to</p>
                    <p className="text-white/50 text-xs sm:text-sm line-clamp-2 sm:line-clamp-none">"{thoughtPortal.node.userInput}"</p>
                  </div>
                )}

                {/* AI's response that spawned this memory */}
                <div className="pt-3 sm:pt-4 border-t border-white/10 mt-3 sm:mt-4">
                  <p className="text-white/30 text-[10px] sm:text-xs uppercase tracking-widest mb-1 sm:mb-2">Kevin's Response</p>
                  <p className="text-white/70 text-xs sm:text-sm leading-relaxed line-clamp-6 sm:line-clamp-none max-w-2xl mx-auto">
                    {thoughtPortal.node.contextText || '(Response not captured for this memory)'}
                  </p>
                </div>

                {/* Zone indicator - responsive */}
                {thoughtPortal.node.zone && thoughtPortal.node.zone !== 'ambient' && (
                  <div className="flex flex-wrap justify-center gap-1.5 sm:gap-2 mt-3 sm:mt-4">
                    <span className={`px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-[10px] sm:text-xs uppercase tracking-wider border ${
                      thoughtPortal.node.zone === 'family' ? 'text-orange-300/80 border-orange-400/30 bg-orange-500/10' :
                      thoughtPortal.node.zone === 'technology' ? 'text-cyan-300/80 border-cyan-400/30 bg-cyan-500/10' :
                      thoughtPortal.node.zone === 'consciousness' ? 'text-purple-300/80 border-purple-400/30 bg-purple-500/10' :
                      'text-amber-300/80 border-amber-400/30 bg-amber-500/10'
                    }`}>
                      {thoughtPortal.node.zone === 'family' ? '♥ Family' :
                       thoughtPortal.node.zone === 'technology' ? '⚡ Technology' :
                       thoughtPortal.node.zone === 'consciousness' ? '✧ Consciousness' :
                       '◆ Projects'}
                    </span>
                    {thoughtPortal.node.aspectRatio && (
                      <span className="px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-[10px] sm:text-xs text-white/40 border border-white/10">
                        {thoughtPortal.node.aspectRatio}
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Close hint - responsive */}
              <div className="mt-4 sm:mt-6 md:mt-8 text-center">
                <p className="text-white/20 text-[10px] sm:text-xs uppercase tracking-widest">
                  Tap anywhere to close
                </p>
              </div>
            </div>
          </div>

          {/* Action buttons - Share and Close */}
          <div className="fixed top-3 right-3 sm:top-6 sm:right-6 flex items-center gap-2 z-20" style={{ marginTop: 'env(safe-area-inset-top)' }}>
            {/* Share button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                shareMemory({
                  title: thoughtPortal.node.shortLabel || 'A Visual Memory',
                  text: thoughtPortal.node.fullPrompt || thoughtPortal.node.contextText || 'A thought from Kevin\'s Digital Mind',
                  imageUrl: thoughtPortal.imageUrl
                });
              }}
              className="p-2 sm:p-3 rounded-full bg-black/60 backdrop-blur-sm border border-white/20 text-white/70 hover:text-white hover:bg-black/80 transition-all"
              title="Share this memory"
            >
              <svg className="w-5 h-5 sm:w-6 sm:h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="18" cy="5" r="3"/>
                <circle cx="6" cy="12" r="3"/>
                <circle cx="18" cy="19" r="3"/>
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
              </svg>
            </button>

            {/* Close button */}
            <button
              onClick={() => setThoughtPortal(null)}
              className="p-2 sm:p-3 rounded-full bg-black/60 backdrop-blur-sm border border-white/20 text-white/70 hover:text-white hover:bg-black/80 transition-all"
            >
              <svg className="w-5 h-5 sm:w-6 sm:h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* ============================================
          UNIFIED CHAT INTERFACE
          One component for all screens with 3 states:
          1. Welcome: Centered input with title (no messages)
          2. Active: Centered expanded chat panel (during conversation)
          3. Collapsed: Input bar only with expand option
          ============================================ */}

      {/* Main Chat Container - Draggable */}
      {(
        <div
          className="absolute top-12 sm:top-4 left-1/2 z-30 pointer-events-none px-3 sm:px-4 pt-[env(safe-area-inset-top)] w-full max-w-2xl"
          style={{
            transform: `translate(calc(-50% + ${chatPosition.x}px), ${chatPosition.y}px)`,
            transition: isDragging ? 'none' : 'transform 0.15s ease-out'
          }}
        >
          <div
            className={`pointer-events-auto w-full transition-all duration-500 ease-out ${
              messages.length > 0 || streamingResponse
                ? 'max-w-2xl'
                : 'max-w-lg mx-auto'
            }`}
          >
            {/* Chat Panel Container - Click anywhere to expand when collapsed */}
            <div
              ref={chatPanelRef}
              onMouseDown={(e) => {
                // Stop propagation to prevent OrbitControls from receiving this event
                e.stopPropagation();
              }}
              onTouchStart={(e) => {
                // Stop propagation to prevent OrbitControls from receiving this event
                e.stopPropagation();
              }}
              onClick={() => {
                // Expand on click if collapsed (and wasn't a drag)
                if (!wasDragRef.current && isChatCollapsed) {
                  expandChat();
                }
                wasDragRef.current = false;
              }}
              onTouchEnd={() => {
                // Handle touch tap to expand (mobile)
                if (!wasDragRef.current && isChatCollapsed) {
                  expandChat();
                }
                wasDragRef.current = false;
              }}
              className={`relative bg-black/40 backdrop-blur-sm rounded-3xl border border-white/10 overflow-hidden shadow-2xl transition-all duration-500 ${
                messages.length > 0 || streamingResponse
                  ? 'shadow-cyan-500/10'
                  : 'shadow-black/50'
              } ${isChatCollapsed ? 'cursor-pointer' : ''}`}
              style={{ boxShadow: '0 25px 80px rgba(0, 0, 0, 0.3)' }}
            >
              {/* Drag Handle - Always visible, drag to move, click to expand if collapsed */}
              <div
                className="flex justify-center py-2 cursor-grab active:cursor-grabbing select-none"
                onMouseDown={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                  wasDragRef.current = false;
                  dragStartRef.current = {
                    x: e.clientX,
                    y: e.clientY,
                    chatX: chatPosition.x,
                    chatY: chatPosition.y
                  };
                }}
                onTouchStart={(e) => {
                  setIsDragging(true);
                  wasDragRef.current = false;
                  dragStartRef.current = {
                    x: e.touches[0].clientX,
                    y: e.touches[0].clientY,
                    chatX: chatPosition.x,
                    chatY: chatPosition.y
                  };
                }}
                onClick={() => {
                  // Only expand if it wasn't a drag
                  if (!wasDragRef.current && isChatCollapsed) {
                    expandChat();
                  }
                }}
              >
                <div className="w-12 h-1 rounded-full bg-white/20 hover:bg-white/40 transition-colors" />
              </div>

              {/* Header - Only shown when there are messages and not collapsed */}
              {(messages.length > 0 || streamingResponse) && !isChatCollapsed && (
                <div className="px-5 py-2 border-b border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-cyan-400/60 animate-pulse" />
                    <span className="text-xs uppercase tracking-[0.15em] text-white/40">Mind of Kevin • Digital Twin</span>
                  </div>
                  <button
                    onClick={() => setIsChatCollapsed(true)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-white/40 hover:text-white/60"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M18 15l-6-6-6 6"/>
                    </svg>
                    <span className="text-xs">Collapse</span>
                  </button>
                </div>
              )}

              {/* Collapsed State - Show message count indicator */}
              {isChatCollapsed && (messages.length > 0 || streamingResponse) && (
                <div
                  className="px-4 py-1 flex items-center justify-center gap-2 text-white/50 cursor-pointer hover:text-white/70 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    expandChat();
                  }}
                >
                  <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="text-xs">{messages.length} messages — tap to expand</span>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M6 9l6 6 6-6"/>
                  </svg>
                </div>
              )}

              {/* Welcome Title & Starter Prompts - Only when no messages and not collapsed */}
              {messages.length === 0 && !streamingResponse && !isChatCollapsed && (
                <div className="text-center py-6 px-6">
                  <p className="text-white text-2xl sm:text-3xl tracking-[0.2em] sm:tracking-[0.3em] uppercase font-light">Mind of Kevin</p>
                  <p className="text-white/90 text-sm sm:text-base mt-1 tracking-[0.15em] uppercase">Digital Twin</p>
                  <p className="text-white/70 text-xs sm:text-sm mt-2 tracking-widest">Speak or type to begin</p>
                  <button
                    onClick={() => { setCurrentQuote(getRandomQuote()); setIsAboutOpen(true); }}
                    className="text-white/80 hover:text-white text-sm mt-4 underline underline-offset-4 transition-colors"
                  >
                    What is this?
                  </button>

                  {/* Starter prompts */}
                  <div className="flex flex-wrap justify-center gap-2 mt-5">
                    {[
                      "Who is Kevin?",
                      "Thoughts on AI",
                      "Latest thoughts"
                    ].map((prompt) => (
                      <button
                        key={prompt}
                        onClick={async () => {
                          // Auto-send the prompt immediately
                          if (isThinking) return;

                          setIsThinking(true);
                          addMessage('user', prompt);
                          lastUserInputRef.current = prompt; // Store for memory context

                          // Visual feedback
                          stateRef.current.targetZone = classifyTopic(prompt);
                          setCurrentZone(stateRef.current.targetZone);
                          stateRef.current.isManual = false;
                          setManualControl(false);

                          try {
                            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
                            const response = await ai.models.generateContent({
                              model: 'gemini-3-pro-preview',
                              contents: [{ role: 'user', parts: [{ text: prompt }] }],
                              config: { systemInstruction: SYSTEM_INSTRUCTION }
                            });
                            const text = response.text || "Thinking...";
                            handleResponse(text);
                          } catch (err) {
                            console.error(err);
                            addMessage('assistant', "Thought stream interrupted.");
                          } finally {
                            setIsThinking(false);
                          }
                        }}
                        disabled={isThinking}
                        className="px-3 py-1.5 text-xs text-white/40 hover:text-white/70 border border-white/10 hover:border-white/25 rounded-full transition-all hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Messages Area - Hidden when collapsed, expands dynamically up to 70vh */}
              {(messages.length > 0 || streamingResponse) && !isChatCollapsed && (
                <div className="max-h-[60vh] sm:max-h-[70vh] overflow-y-auto custom-scrollbar p-3 sm:p-5 overscroll-contain touch-pan-y">
                  <div className="flex flex-col gap-4">
                    {messages.map((msg) => (
                      <div key={msg.id}>
                        {/* Image message */}
                        {msg.role === 'images' && msg.images && (
                          <div className="flex justify-center">
                            <div className="flex gap-2 p-3 rounded-2xl bg-white/[0.04] border border-white/5">
                              {msg.images.map((img, idx) => {
                                const queueIndex = (msg.queueStartIndex ?? 0) + idx;
                                return (
                                  <div
                                    key={idx}
                                    className="relative group cursor-pointer"
                                    onClick={() => {
                                      const S = stateRef.current;
                                      const node = S.cameraFocusNodes[queueIndex];
                                      if (node && node.mesh) {
                                        const worldPos = new THREE.Vector3();
                                        node.mesh.getWorldPosition(worldPos);
                                        S.cameraFocusTarget = worldPos;
                                        S.cameraFocusNodeIndex = queueIndex;
                                        S.cameraOrbitPhase = 0;
                                        S.isManual = false;
                                        setManualControl(false);
                                        setIsChatCollapsed(true);
                                        setIsNavigatingToImages(true);
                                      }
                                    }}
                                  >
                                    <img
                                      src={img}
                                      alt={`Generated scene ${idx + 1}`}
                                      className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl object-cover opacity-90 group-hover:opacity-100 transition-all group-hover:scale-105 group-hover:shadow-lg group-hover:shadow-cyan-500/20"
                                    />
                                    <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                        {/* Visualization offer */}
                        {msg.role === 'visualize_offer' && msg.offerContext && (
                          <div className="flex justify-start animate-fade-in">
                            <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md bg-gradient-to-br from-violet-500/8 via-cyan-500/5 to-indigo-500/8 border border-violet-500/15 hover:border-violet-500/25 transition-colors">
                              <div className="flex items-center gap-3">
                                {/* Eye icon - subtle and elegant */}
                                <div className="relative w-9 h-9 flex-shrink-0">
                                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-400/15 to-cyan-400/10" />
                                  <div className="absolute inset-1 rounded-full bg-black/30 flex items-center justify-center">
                                    <svg className="w-4 h-4 text-violet-300/60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                                      <circle cx="12" cy="12" r="3" />
                                    </svg>
                                  </div>
                                </div>
                                <p className="text-violet-200/70 text-sm">Would you like to see what I'm visualizing?</p>
                              </div>

                              {/* Action buttons */}
                              <div className="flex gap-2 mt-3">
                                <button
                                  onClick={() => handleAcceptVisualization(msg.id, msg.offerContext!.responseText, msg.offerContext!.topic)}
                                  className="flex-1 px-3 py-1.5 text-xs font-medium text-violet-200 bg-violet-500/15 hover:bg-violet-500/25 border border-violet-500/20 hover:border-violet-500/35 rounded-lg transition-all active:scale-[0.98]"
                                >
                                  Show me
                                </button>
                                <button
                                  onClick={() => handleDismissVisualization(msg.id)}
                                  className="px-3 py-1.5 text-xs text-white/40 hover:text-white/60 border border-white/10 hover:border-white/20 rounded-lg transition-all active:scale-[0.98]"
                                >
                                  Skip
                                </button>
                              </div>
                            </div>
                          </div>
                        )}
                        {/* Text messages */}
                        {msg.role !== 'images' && msg.role !== 'visualize_offer' && (
                          <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div
                              className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm sm:text-base leading-relaxed
                              ${msg.role === 'user'
                                ? 'bg-gradient-to-br from-cyan-500/25 to-blue-600/25 text-white border border-cyan-500/20 rounded-br-md'
                                : 'bg-white/[0.08] text-white/90 border border-white/5 rounded-bl-md'
                              }`}
                              style={{
                                boxShadow: msg.role === 'user'
                                  ? '0 4px 20px rgba(6, 182, 212, 0.15)'
                                  : '0 4px 20px rgba(0, 0, 0, 0.3)'
                              }}
                            >
                              {msg.content}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                    {/* Thinking indicator - shows while waiting for AI response */}
                    {isThinking && !streamingResponse && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md bg-white/[0.06] border border-white/10">
                          <div className="flex items-center gap-3">
                            {/* Animated brain/thinking icon */}
                            <div className="relative w-8 h-8 flex-shrink-0">
                              {/* Pulsing glow */}
                              <div className="absolute inset-0 rounded-full bg-cyan-500/20 animate-pulse" />
                              {/* Inner circle with dots */}
                              <div className="absolute inset-0 rounded-full bg-black/30 flex items-center justify-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/80 animate-bounce" style={{ animationDelay: '0ms', animationDuration: '1s' }} />
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/80 animate-bounce" style={{ animationDelay: '150ms', animationDuration: '1s' }} />
                                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/80 animate-bounce" style={{ animationDelay: '300ms', animationDuration: '1s' }} />
                              </div>
                            </div>
                            <p className="text-white/50 text-sm">Thinking...</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Imagining visual memory indicator */}
                    {isImagining && (
                      <div className="flex justify-start animate-fade-in">
                        <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md bg-gradient-to-br from-violet-500/10 via-cyan-500/5 to-purple-500/10 border border-violet-500/20">
                          <div className="flex items-center gap-3">
                            {/* Animated thought bubble / eye icon */}
                            <div className="relative w-10 h-10 flex-shrink-0">
                              {/* Outer glow ring */}
                              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-400/30 to-cyan-400/30 animate-pulse" />
                              {/* Inner circle with eye */}
                              <div className="absolute inset-1 rounded-full bg-black/40 flex items-center justify-center">
                                <svg className="w-5 h-5 text-violet-300/80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                                  <circle cx="12" cy="12" r="3" />
                                </svg>
                              </div>
                              {/* Orbiting particles */}
                              <div className="absolute inset-0 animate-spin" style={{ animationDuration: '3s' }}>
                                <div className="absolute top-0 left-1/2 w-1 h-1 -ml-0.5 rounded-full bg-cyan-400/60" />
                              </div>
                              <div className="absolute inset-0 animate-spin" style={{ animationDuration: '4s', animationDirection: 'reverse' }}>
                                <div className="absolute bottom-0 left-1/2 w-1 h-1 -ml-0.5 rounded-full bg-violet-400/60" />
                              </div>
                            </div>

                            <div className="flex-1 min-w-0">
                              <p className="text-violet-200/80 text-sm font-medium">Creating visual memory...</p>
                              <p className="text-white/40 text-xs mt-0.5 truncate italic">{imaginingLabel}</p>
                            </div>
                          </div>

                          {/* Shimmer loading bar */}
                          <div className="mt-3 h-1 rounded-full bg-white/5 overflow-hidden">
                            <div
                              className="h-full w-1/3 rounded-full bg-gradient-to-r from-transparent via-violet-400/50 to-transparent"
                              style={{
                                animation: 'shimmer 2s ease-in-out infinite',
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Streaming response */}
                    {streamingResponse && (
                      <div className="flex justify-start">
                        <div
                          className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md text-sm sm:text-base leading-relaxed bg-white/[0.08] text-white/90 border border-white/5"
                          style={{ boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)' }}
                        >
                          {streamingResponse}
                          <span className="inline-block w-0.5 h-4 ml-1 align-middle bg-cyan-400 animate-blink" />
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </div>
              )}

              {/* Input Area - Always visible */}
              <div className={`p-4 ${(messages.length > 0 || streamingResponse) && !isChatCollapsed ? 'border-t border-white/5' : ''}`}>
                <form onSubmit={handleChatSubmit} className="relative group w-full">
                  {/* Ambient glow behind input */}
                  <div
                    className="absolute -inset-1 rounded-full bg-gradient-to-r from-cyan-500/20 via-purple-500/10 to-cyan-500/20 blur-xl opacity-50 group-focus-within:opacity-80 transition-opacity duration-500"
                    style={{ filter: 'blur(20px)' }}
                  />

                  <div className="relative flex items-center gap-3">
                    {/* Voice button - 44px minimum touch target for accessibility */}
                    {!isConnected ? (
                      <button
                        type="button"
                        onClick={connect}
                        className="flex-shrink-0 group/mic flex items-center justify-center w-11 h-11 min-w-[44px] min-h-[44px] rounded-full bg-black/60 backdrop-blur-xl border border-white/15 hover:bg-black/70 hover:border-white/25 active:scale-95 transition-all"
                        style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)' }}
                        title="Start Voice"
                      >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50 group-hover/mic:text-white/90 transition-colors">
                          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                          <line x1="12" y1="19" x2="12" y2="23"/>
                          <line x1="8" y1="23" x2="16" y2="23"/>
                        </svg>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={disconnect}
                        className="flex-shrink-0 group/mic relative flex items-center justify-center w-11 h-11 min-w-[44px] min-h-[44px] rounded-full bg-black/60 backdrop-blur-xl border border-cyan-500/30 hover:bg-black/70 active:scale-95 transition-all"
                        style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(6, 182, 212, 0.15)' }}
                        title="End Voice"
                      >
                        {isSpeaking && (
                          <>
                            <div className="absolute inset-0 rounded-full border-2 border-cyan-400/40 animate-ping" />
                            <div className="absolute inset-1 rounded-full border border-cyan-400/20 animate-pulse" />
                          </>
                        )}
                        <div className="w-3 h-3 bg-cyan-400 rounded-full shadow-[0_0_15px_rgba(34,211,238,0.6)]" />
                      </button>
                    )}

                    {/* Input field */}
                    <div
                      className="relative flex-1"
                      onClick={() => {
                        // Expand chat when clicking on input area (if collapsed)
                        if (isChatCollapsed) {
                          expandChat();
                        }
                      }}
                    >
                      <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onMouseDown={() => {
                          // Expand IMMEDIATELY on mouse down (before focus steals click)
                          wasDragRef.current = false;
                          if (isChatCollapsed) {
                            expandChat();
                          }
                        }}
                        onTouchStart={() => {
                          // Handle touch tap on input (mobile) - fires before touchEnd
                          wasDragRef.current = false;
                          if (isChatCollapsed) {
                            expandChat();
                          }
                        }}
                        onFocus={() => {
                          // Also expand on focus (keyboard navigation, etc.)
                          wasDragRef.current = false;
                          if (isChatCollapsed) {
                            expandChat();
                          }
                        }}
                        placeholder={isConnected ? "listening..." : "What's on your mind?"}
                        className="w-full bg-black/60 backdrop-blur-xl border border-white/15 rounded-full py-3 pl-4 pr-12 text-white text-base placeholder-white/40 focus:outline-none focus:bg-black/70 focus:border-white/30 focus:shadow-lg focus:shadow-cyan-500/10 transition-all"
                        style={{
                          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
                          fontSize: '16px', // Prevents iOS zoom on focus
                        }}
                        disabled={isThinking}
                        autoComplete="off"
                        autoCorrect="off"
                        spellCheck="false"
                      />
                      <button
                        type="submit"
                        disabled={!inputText.trim() || isThinking}
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 w-9 h-9 min-w-[36px] min-h-[36px] rounded-full flex items-center justify-center bg-gradient-to-br from-cyan-500/30 to-blue-600/30 text-white/70 hover:text-white hover:from-cyan-500/50 hover:to-blue-600/50 active:scale-95 disabled:opacity-20 disabled:hover:from-cyan-500/30 disabled:hover:to-blue-600/30 transition-all border border-white/10"
                      >
                        {isThinking ? (
                          <div className="w-4 h-4 sm:w-5 sm:h-5 border-2 border-white/40 border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="sm:w-[18px] sm:h-[18px]">
                            <path d="M5 12h14M12 5l7 7-7 7"/>
                          </svg>
                        )}
                      </button>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Manual Control Reset */}
      {manualControl && (
        <div className="absolute top-4 left-4 sm:top-6 sm:left-6 z-30 pointer-events-auto animate-fade-in">
          <button
            onClick={handleResetView}
            className="px-3 py-1.5 bg-black/30 hover:bg-black/50 border border-white/10 rounded-full text-[10px] uppercase tracking-widest text-white/40 hover:text-white/70 backdrop-blur-sm transition-all flex items-center gap-2"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-pulse" />
            <span>free roam</span>
            <span className="text-white/20">×</span>
          </button>
        </div>
      )}

      {/* Zone Label - hidden on mobile, shown on desktop */}
      {currentZone !== 'ambient' && (
        <div className="hidden sm:block absolute bottom-20 left-6 pointer-events-none z-0">
          <p className="text-[9px] tracking-[0.4em] uppercase opacity-20 transition-colors duration-1000"
             style={{
               color: currentZone === 'technology' ? '#00ffff' :
                      currentZone === 'family' ? '#ffaa44' :
                      currentZone === 'consciousness' ? '#aa66ff' : '#44aaff'
             }}>
            {zoneLabels[currentZone]}
          </p>
        </div>
      )}

      {/* Floating Navigation Orbs - centered horizontally at bottom */}
      <div
        className={`absolute z-20 pointer-events-auto left-1/2 -translate-x-1/2 bottom-28 sm:bottom-8 flex flex-row gap-4 sm:gap-6 transition-all duration-500 ${
          !isChatCollapsed && (messages.length > 0 || streamingResponse) ? 'opacity-0 pointer-events-none translate-y-4' : 'opacity-100 translate-y-0'
        }`}
      >
        {/* Memory Gallery - 44px touch target */}
        <button
          onClick={() => setIsGalleryOpen(true)}
          className={`group relative flex items-center justify-center transition-all duration-500 min-w-[44px] min-h-[44px] ${savedMemories.length > 0 ? 'opacity-60 hover:opacity-100' : 'opacity-40 hover:opacity-70'} active:opacity-100`}
        >
          <span className="hidden sm:block absolute bottom-12 text-xs tracking-[0.12em] uppercase text-white/70 whitespace-nowrap transition-all duration-300 opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0">
            {savedMemories.length} Memories
          </span>
          <div className="relative w-6 h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 active:scale-90 bg-white/40 sm:group-hover:bg-white/60 sm:group-hover:shadow-[0_0_20px_rgba(255,255,255,0.4)]">
            <div className="absolute inset-0 rounded-full bg-white/30 sm:group-hover:animate-pulse" style={{ animationDuration: '2s' }} />
            <svg className="absolute inset-0 w-full h-full p-1.5 text-white/80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
          </div>
        </button>

        {/* The Horizon - Speaking/Ideas - 44px touch target */}
        <button
          onClick={() => setActivePanel(activePanel === 'horizon' ? null : 'horizon')}
          className={`group relative flex items-center justify-center transition-all duration-500 min-w-[44px] min-h-[44px] ${activePanel === 'horizon' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute bottom-12 text-sm tracking-[0.15em] uppercase text-amber-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'horizon' ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0'}`}>
            The Horizon
          </span>
          <div className={`relative w-6 h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 active:scale-90 ${activePanel === 'horizon' ? 'bg-amber-400 shadow-[0_0_30px_rgba(251,191,36,0.7)] scale-125' : 'bg-amber-400/70 sm:group-hover:bg-amber-400 sm:group-hover:shadow-[0_0_25px_rgba(251,191,36,0.5)]'}`}>
            <div className={`absolute inset-0 rounded-full bg-amber-400/50 ${activePanel === 'horizon' ? 'animate-ping' : 'sm:group-hover:animate-pulse'}`} style={{ animationDuration: '2s' }} />
          </div>
        </button>

        {/* The Bridge - Connection - 44px touch target */}
        <button
          onClick={() => setActivePanel(activePanel === 'bridge' ? null : 'bridge')}
          className={`group relative flex items-center justify-center transition-all duration-500 min-w-[44px] min-h-[44px] ${activePanel === 'bridge' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute bottom-12 text-sm tracking-[0.15em] uppercase text-cyan-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'bridge' ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0'}`}>
            The Bridge
          </span>
          <div className={`relative w-6 h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 active:scale-90 ${activePanel === 'bridge' ? 'bg-cyan-400 shadow-[0_0_30px_rgba(34,211,238,0.7)] scale-125' : 'bg-cyan-400/70 sm:group-hover:bg-cyan-400 sm:group-hover:shadow-[0_0_25px_rgba(34,211,238,0.5)]'}`}>
            <div className={`absolute inset-0 rounded-full bg-cyan-400/50 ${activePanel === 'bridge' ? 'animate-ping' : 'sm:group-hover:animate-pulse'}`} style={{ animationDuration: '2s' }} />
          </div>
        </button>

        {/* The Echoes - Writings - 44px touch target */}
        <button
          onClick={() => setActivePanel(activePanel === 'echoes' ? null : 'echoes')}
          className={`group relative flex items-center justify-center transition-all duration-500 min-w-[44px] min-h-[44px] ${activePanel === 'echoes' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute bottom-12 text-sm tracking-[0.15em] uppercase text-violet-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'echoes' ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 group-hover:opacity-100 group-hover:translate-y-0'}`}>
            The Echoes
          </span>
          <div className={`relative w-6 h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 active:scale-90 ${activePanel === 'echoes' ? 'bg-violet-400 shadow-[0_0_30px_rgba(167,139,250,0.7)] scale-125' : 'bg-violet-400/70 sm:group-hover:bg-violet-400 sm:group-hover:shadow-[0_0_25px_rgba(167,139,250,0.5)]'}`}>
            <div className={`absolute inset-0 rounded-full bg-violet-400/50 ${activePanel === 'echoes' ? 'animate-ping' : 'sm:group-hover:animate-pulse'}`} style={{ animationDuration: '2s' }} />
          </div>
        </button>
      </div>

      {/* The Horizon Panel - Speaking & Ideas */}
      {activePanel === 'horizon' && (
        <div className="fixed inset-0 z-40 pointer-events-auto animate-fade-in sm:absolute sm:inset-auto sm:top-1/2 sm:right-20 sm:-translate-y-1/2 sm:z-20">
          {/* Mobile backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm sm:hidden" onClick={() => setActivePanel(null)} />
          <div className="absolute inset-4 sm:inset-auto sm:relative bg-black/90 sm:bg-black/80 backdrop-blur-xl rounded-2xl border border-amber-500/20 p-5 sm:p-6 sm:w-80 shadow-2xl shadow-amber-500/10 flex flex-col max-h-[calc(100vh-2rem)] sm:max-h-none overflow-y-auto">
            {/* Close button - mobile only */}
            <button onClick={() => setActivePanel(null)} className="absolute top-3 right-3 p-2 text-white/40 hover:text-white/70 sm:hidden">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
            </button>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_10px_rgba(251,191,36,0.6)]" />
              <h3 className="text-amber-200 text-sm tracking-[0.15em] uppercase">The Horizon</h3>
            </div>
            <p className="text-white/70 text-sm leading-relaxed mb-5">
              Speaking on AI consciousness, the future of human-machine collaboration, and what it means to teach sand to think.
            </p>
            <div className="space-y-3 relative z-10">
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setCurrentQuote(getRandomQuote()); setActiveHorizonModal('keynote'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setCurrentQuote(getRandomQuote()); setActiveHorizonModal('keynote'); }}
                className="w-full text-left p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors pointer-events-none">Keynote Speaking</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">Conferences, corporate events, universities</span>
              </div>
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveHorizonModal('workshops'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveHorizonModal('workshops'); }}
                className="w-full text-left p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors pointer-events-none">Workshops</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">Interactive sessions on AI strategy & ethics</span>
              </div>
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveHorizonModal('advisory'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveHorizonModal('advisory'); }}
                className="w-full text-left p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors pointer-events-none">Advisory</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">Strategic guidance for organizations</span>
              </div>
            </div>
            <button
              onClick={() => setActivePanel('bridge')}
              className="mt-5 w-full py-3 sm:py-2.5 rounded-xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 text-amber-200 text-sm hover:from-amber-500/30 hover:to-orange-500/30 active:from-amber-500/30 active:to-orange-500/30 transition-all"
            >
              Inquire About Booking
            </button>
          </div>
        </div>
      )}

      {/* The Bridge Panel - Conversational Connection */}
      {activePanel === 'bridge' && (
        <div className="fixed inset-0 z-40 pointer-events-auto animate-fade-in sm:absolute sm:inset-auto sm:top-1/2 sm:right-20 sm:-translate-y-1/2 sm:z-20">
          {/* Mobile backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm sm:hidden" onClick={() => { setActivePanel(null); resetBridge(); }} />
          <div className="absolute inset-4 sm:inset-auto sm:relative bg-black/90 sm:bg-black/80 backdrop-blur-xl rounded-2xl border border-cyan-500/20 p-5 sm:p-6 sm:w-96 shadow-2xl shadow-cyan-500/10 flex flex-col max-h-[calc(100vh-2rem)] sm:max-h-[500px]">
            {/* Close button - mobile only */}
            <button onClick={() => { setActivePanel(null); resetBridge(); }} className="absolute top-3 right-3 p-2 text-white/40 hover:text-white/70 sm:hidden z-10">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
            </button>

            {/* Header */}
            <div className="flex items-center gap-3 mb-4 flex-shrink-0">
              <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.6)]" />
              <h3 className="text-cyan-200 text-sm tracking-[0.15em] uppercase">The Bridge</h3>
            </div>

            {/* Chat Step */}
            {bridgeStep === 'chat' && (
              <>
                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 mb-4 min-h-0">
                  {bridgeMessages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'visitor' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[85%] px-3 py-2 rounded-xl text-sm leading-relaxed ${
                        msg.role === 'visitor'
                          ? 'bg-cyan-500/15 border border-cyan-500/20 text-white/90 rounded-br-md'
                          : 'bg-white/5 border border-white/10 text-white/80 rounded-bl-md'
                      }`}>
                        {msg.content}
                      </div>
                    </div>
                  ))}

                  {/* Thinking indicator */}
                  {bridgeIsThinking && (
                    <div className="flex justify-start">
                      <div className="px-3 py-2 rounded-xl rounded-bl-md bg-white/5 border border-white/10">
                        <div className="flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Input */}
                <form
                  onSubmit={(e) => { e.preventDefault(); handleBridgeSubmit(); }}
                  className="flex-shrink-0 flex gap-2"
                >
                  <input
                    type="text"
                    value={bridgeInput}
                    onChange={(e) => setBridgeInput(e.target.value)}
                    placeholder="Share your thoughts..."
                    disabled={bridgeIsThinking}
                    className="flex-1 px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40 disabled:opacity-50"
                  />
                  <button
                    type="submit"
                    disabled={!bridgeInput.trim() || bridgeIsThinking}
                    className="px-4 py-2.5 rounded-xl bg-cyan-500/20 border border-cyan-500/30 text-cyan-200 hover:bg-cyan-500/30 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </svg>
                  </button>
                </form>
              </>
            )}

            {/* Contact Collection Step */}
            {bridgeStep === 'contact' && (
              <div className="flex-1 flex flex-col animate-fade-in">
                <p className="text-white/60 text-sm mb-4">
                  I'd love to pass this along to Kevin. Where can he reach you?
                </p>
                <form
                  onSubmit={(e) => { e.preventDefault(); submitBridgeConversation(); }}
                  className="space-y-3"
                >
                  <input
                    type="text"
                    value={bridgeContact.name}
                    onChange={(e) => setBridgeContact(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="Your name"
                    className="w-full px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40"
                  />
                  <input
                    type="email"
                    value={bridgeContact.email}
                    onChange={(e) => setBridgeContact(prev => ({ ...prev, email: e.target.value }))}
                    placeholder="Your email"
                    required
                    className="w-full px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40"
                  />
                  <button
                    type="submit"
                    disabled={!bridgeContact.email.trim()}
                    className="w-full py-3 sm:py-2.5 rounded-xl bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 text-cyan-200 text-sm hover:from-cyan-500/30 hover:to-blue-500/30 transition-all disabled:opacity-30"
                  >
                    Send to Kevin
                  </button>
                </form>
              </div>
            )}

            {/* Complete Step */}
            {bridgeStep === 'complete' && (
              <div className="flex-1 flex flex-col items-center justify-center text-center animate-fade-in">
                <div className="w-16 h-16 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center mb-4">
                  <svg className="w-8 h-8 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <h4 className="text-white text-lg mb-2">Message Received</h4>
                <p className="text-white/50 text-sm mb-6">
                  Your thoughts have crossed the bridge. Kevin will be in touch.
                </p>
                <button
                  onClick={() => { setActivePanel(null); resetBridge(); }}
                  className="px-6 py-2 rounded-xl bg-white/5 border border-white/10 text-white/70 text-sm hover:bg-white/10 transition-all"
                >
                  Close
                </button>
              </div>
            )}

            {/* Social Links - only show during chat */}
            {bridgeStep === 'chat' && (
              <div className="mt-4 pt-4 border-t border-white/5 flex justify-center gap-6 sm:gap-4 flex-shrink-0">
                <a href="https://x.com/kevin__russell" target="_blank" rel="noopener noreferrer" className="text-white/30 hover:text-cyan-400 active:text-cyan-400 transition-colors p-2">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="sm:w-[18px] sm:h-[18px]"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                </a>
                <a href="https://kevinrussell.substack.com/" target="_blank" rel="noopener noreferrer" className="text-white/30 hover:text-cyan-400 active:text-cyan-400 transition-colors p-2">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="sm:w-[18px] sm:h-[18px]"><path d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46zM22.54 0H1.46v2.836h21.08V0z"/></svg>
                </a>
              </div>
            )}
          </div>
        </div>
      )}

      {/* The Echoes Panel - Writings */}
      {activePanel === 'echoes' && (
        <div className="fixed inset-0 z-40 pointer-events-auto animate-fade-in sm:absolute sm:inset-auto sm:top-1/2 sm:right-20 sm:-translate-y-1/2 sm:z-20">
          {/* Mobile backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm sm:hidden" onClick={() => setActivePanel(null)} />
          <div className="absolute inset-4 sm:inset-auto sm:relative bg-black/90 sm:bg-black/80 backdrop-blur-xl rounded-2xl border border-violet-500/20 p-5 sm:p-6 sm:w-80 shadow-2xl shadow-violet-500/10 flex flex-col max-h-[calc(100vh-2rem)] sm:max-h-none overflow-y-auto">
            {/* Close button - mobile only */}
            <button onClick={() => setActivePanel(null)} className="absolute top-3 right-3 p-2 text-white/40 hover:text-white/70 sm:hidden">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
            </button>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-violet-400 shadow-[0_0_10px_rgba(167,139,250,0.6)]" />
              <h3 className="text-violet-200 text-sm tracking-[0.15em] uppercase">The Echoes</h3>
            </div>
            <p className="text-white/70 text-sm leading-relaxed mb-5">
              Thoughts crystallized. Words that linger.
            </p>
            <div className="space-y-3 relative z-10">
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('sand-speaks'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('sand-speaks'); }}
                className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group cursor-pointer select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors pointer-events-none">When Sand Speaks</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">A meditation on AI consciousness</span>
              </div>
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('emma-project'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('emma-project'); }}
                className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group cursor-pointer select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors pointer-events-none">The Emma Project</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">Preserving memory, honoring legacy</span>
              </div>
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('essays'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('essays'); }}
                className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group cursor-pointer select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors pointer-events-none">Essays & Reflections</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">On technology, family, and meaning</span>
              </div>
              <div
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('books'); }}
                onTouchEnd={(e) => { e.preventDefault(); e.stopPropagation(); setActiveEchoesModal('books'); }}
                className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group cursor-pointer select-none"
                role="button"
                tabIndex={0}
              >
                <span className="block text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors pointer-events-none">Books</span>
                <span className="block text-white/40 text-xs mt-1 pointer-events-none">Written works by Kevin Russell</span>
              </div>
            </div>
            <div className="mt-5 p-3 sm:p-3 rounded-xl bg-gradient-to-r from-violet-500/10 to-purple-500/10 border border-violet-500/20">
              <p className="text-violet-200/80 text-xs italic leading-relaxed">
                "We are clever apes who taught sand to think. Now we must teach ourselves what that means."
              </p>
              <p className="text-white/30 text-[10px] mt-2">- Kevin Russell</p>
            </div>
          </div>
        </div>
      )}

      {/* ============ HORIZON MODALS ============ */}

      {/* Keynote Speaking Modal */}
      {activeHorizonModal === 'keynote' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveHorizonModal(null)}
        >
          {/* Backdrop with amber glow */}
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-amber-900/20 via-transparent to-orange-900/20" />

          {/* Floating particles effect */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute w-1 h-1 bg-amber-400/30 rounded-full animate-pulse"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`,
                  animationDuration: `${2 + Math.random() * 3}s`
                }}
              />
            ))}
          </div>

          {/* Modal Content */}
          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-amber-950/30 backdrop-blur-xl rounded-3xl border border-amber-500/20 shadow-2xl shadow-amber-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header with animated gradient */}
            <div className="relative p-8 pb-6 border-b border-amber-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500/5 via-orange-500/5 to-amber-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveHorizonModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-amber-400">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">Keynote Speaking</h2>
                  <p className="text-amber-200/60 text-sm mt-1 tracking-wide">Illuminating the path forward</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Dynamic Quote */}
              <div className="relative pl-6 border-l-2 border-amber-500/30">
                <p className="text-white/80 text-lg italic leading-relaxed">
                  "{currentQuote.text}"
                </p>
                <p className="text-amber-200/50 text-sm mt-3">— {currentQuote.author}</p>
              </div>

              {/* Topics Section - Clickable to explore each topic */}
              <div>
                <h3 className="text-amber-200 text-sm tracking-[0.2em] uppercase mb-4">Speaking Topics</h3>
                <p className="text-white/40 text-xs mb-4">Click any topic to explore it deeper</p>
                <div className="grid gap-3">
                  {SPEAKING_TOPICS.map((topic, i) => (
                    <div
                      key={topic.id}
                      onClick={() => setSelectedTopic(topic)}
                      className="p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:border-amber-500/30 hover:bg-amber-500/5 transition-all duration-300 group cursor-pointer"
                    >
                      <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0 group-hover:bg-amber-500/20 group-hover:scale-110 transition-all">
                          <span className="text-lg">{topic.icon}</span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <p className="text-white/90 font-medium group-hover:text-amber-200 transition-colors">{topic.title}</p>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/20 group-hover:text-amber-400 group-hover:translate-x-1 transition-all">
                              <path d="M9 18l6-6-6-6"/>
                            </svg>
                          </div>
                          <p className="text-white/40 text-sm mt-1">{topic.tagline}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* CTA */}
              <div className="pt-4 border-t border-white/5">
                <button
                  onClick={() => { setActiveHorizonModal(null); setActivePanel('bridge'); }}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-amber-500/20 border border-amber-500/30 text-amber-200 font-medium tracking-wide hover:from-amber-500/30 hover:via-orange-500/30 hover:to-amber-500/30 transition-all duration-300 group"
                >
                  <span className="flex items-center justify-center gap-3">
                    Book a Speaking Engagement
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:translate-x-1 transition-transform">
                      <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Topic Detail Modal - Expanded view of a speaking topic */}
      {selectedTopic && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setSelectedTopic(null)}
        >
          <div className="absolute inset-0 bg-black/90 backdrop-blur-xl" />
          <div className="absolute inset-0 bg-gradient-to-br from-amber-900/20 via-transparent to-orange-900/20" />

          {/* Animated background particles */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute w-1 h-1 bg-amber-400/30 rounded-full animate-pulse"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 3}s`,
                  animationDuration: `${2 + Math.random() * 3}s`
                }}
              />
            ))}
          </div>

          {/* Modal Content */}
          <div
            className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/98 via-black/95 to-amber-950/20 backdrop-blur-xl rounded-3xl border border-amber-500/20 shadow-2xl shadow-amber-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header with Icon */}
            <div className="relative p-8 pb-6 border-b border-amber-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500/5 via-orange-500/5 to-amber-500/5" />
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-amber-500/50 to-transparent" />

              <button
                onClick={() => setSelectedTopic(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-start gap-5">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center shadow-lg shadow-amber-500/20">
                  <span className="text-4xl">{selectedTopic.icon}</span>
                </div>
                <div className="flex-1 pt-1">
                  <h2 className="text-3xl font-light text-white tracking-wide">{selectedTopic.title}</h2>
                  <p className="text-amber-200/60 text-lg mt-2 italic">{selectedTopic.tagline}</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Description */}
              <div>
                <p className="text-white/80 text-lg leading-relaxed">
                  {selectedTopic.description}
                </p>
              </div>

              {/* Key Points */}
              <div className="space-y-4">
                <h3 className="text-amber-200 text-sm tracking-[0.2em] uppercase flex items-center gap-2">
                  <span className="w-8 h-[1px] bg-amber-500/30" />
                  What We Explore
                </h3>
                <div className="grid gap-3">
                  {selectedTopic.keyPoints.map((point, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-4 p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-amber-500/20 transition-all duration-300"
                    >
                      <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                        <span className="text-amber-400 font-mono text-sm">{String(i + 1).padStart(2, '0')}</span>
                      </div>
                      <p className="text-white/70 pt-1">{point}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Audience Gets */}
              <div className="p-6 rounded-2xl bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20">
                <h3 className="text-amber-200 text-sm tracking-[0.2em] uppercase mb-3 flex items-center gap-2">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-amber-400">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                  </svg>
                  What You Take Away
                </h3>
                <p className="text-white/80 leading-relaxed">{selectedTopic.audienceGets}</p>
              </div>

              {/* CTA Section */}
              <div className="pt-6 border-t border-white/5 space-y-4">
                <p className="text-center text-white/40 text-sm">
                  Want to explore this topic in real-time?
                </p>

                {/* Discuss with Twin Button - THE WOW FACTOR */}
                <button
                  onClick={() => startTopicConversation(selectedTopic)}
                  className="w-full py-5 rounded-2xl bg-gradient-to-r from-amber-500/30 via-orange-500/30 to-amber-500/30 border border-amber-500/40 text-amber-100 font-medium tracking-wide hover:from-amber-500/40 hover:via-orange-500/40 hover:to-amber-500/40 hover:border-amber-400/50 transition-all duration-500 group relative overflow-hidden"
                >
                  {/* Shimmer effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />

                  <span className="relative flex items-center justify-center gap-3">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-amber-300 group-hover:scale-110 transition-transform">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    <span className="text-lg">Discuss This with Kevin's Twin</span>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:translate-x-2 transition-transform">
                      <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                  </span>
                </button>

                <p className="text-center text-white/30 text-xs">
                  Start an AI-powered conversation exploring this topic together
                </p>

                {/* Secondary Actions */}
                <div className="flex gap-3 pt-2">
                  <button
                    onClick={() => { setSelectedTopic(null); }}
                    className="flex-1 py-3 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm hover:bg-white/10 hover:text-white/80 transition-all"
                  >
                    Back to Topics
                  </button>
                  <button
                    onClick={() => { setSelectedTopic(null); setActiveHorizonModal(null); setActivePanel('bridge'); }}
                    className="flex-1 py-3 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm hover:bg-white/10 hover:text-white/80 transition-all"
                  >
                    Book This Talk
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Workshops Modal */}
      {activeHorizonModal === 'workshops' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveHorizonModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/20 via-transparent to-teal-900/20" />

          {/* Modal Content */}
          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-emerald-950/30 backdrop-blur-xl rounded-3xl border border-emerald-500/20 shadow-2xl shadow-emerald-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-emerald-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 via-teal-500/5 to-emerald-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveHorizonModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-emerald-400">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                    <circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">Interactive Workshops</h2>
                  <p className="text-emerald-200/60 text-sm mt-1 tracking-wide">Hands-on learning experiences</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Description */}
              <p className="text-white/70 leading-relaxed">
                Immersive, hands-on sessions designed to bridge the gap between understanding AI concepts
                and implementing them in real-world scenarios. Each workshop combines philosophical inquiry
                with practical application.
              </p>

              {/* Workshop Types */}
              <div className="space-y-4">
                <h3 className="text-emerald-200 text-sm tracking-[0.2em] uppercase mb-4">Workshop Formats</h3>

                {[
                  {
                    title: 'AI Strategy Intensive',
                    duration: 'Full Day',
                    capacity: '20-50 participants',
                    desc: 'Chart your organization\'s AI journey with frameworks for ethical implementation and competitive advantage.',
                    topics: ['Strategic Assessment', 'Implementation Roadmap', 'Risk Mitigation', 'Team Alignment']
                  },
                  {
                    title: 'Ethics in AI Development',
                    duration: 'Half Day',
                    capacity: '15-30 participants',
                    desc: 'Navigate the moral landscape of artificial intelligence with practical ethical frameworks.',
                    topics: ['Bias Detection', 'Transparency Principles', 'Accountability Models', 'Case Studies']
                  },
                  {
                    title: 'Future of Work Symposium',
                    duration: '2-3 Hours',
                    capacity: '30-100 participants',
                    desc: 'Prepare your team for human-AI collaboration in the evolving workplace.',
                    topics: ['Augmentation vs Automation', 'Skill Evolution', 'Organizational Design', 'Change Management']
                  }
                ].map((workshop, i) => (
                  <div
                    key={i}
                    className="p-5 rounded-2xl bg-white/[0.03] border border-white/5 hover:border-emerald-500/20 transition-all duration-300"
                  >
                    <div className="flex items-start justify-between gap-4 mb-3">
                      <h4 className="text-white/90 font-medium text-lg">{workshop.title}</h4>
                      <div className="flex gap-2 flex-shrink-0">
                        <span className="px-2 py-1 rounded-md bg-emerald-500/10 text-emerald-300 text-xs">{workshop.duration}</span>
                      </div>
                    </div>
                    <p className="text-white/50 text-sm mb-3">{workshop.desc}</p>
                    <div className="flex flex-wrap gap-2">
                      {workshop.topics.map((topic, j) => (
                        <span key={j} className="px-2 py-1 rounded-md bg-white/5 text-white/40 text-xs">{topic}</span>
                      ))}
                    </div>
                    <p className="text-emerald-200/40 text-xs mt-3">{workshop.capacity}</p>
                  </div>
                ))}
              </div>

              {/* What's Included */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-emerald-500/5 to-teal-500/5 border border-emerald-500/10">
                <h3 className="text-emerald-200 text-sm tracking-[0.2em] uppercase mb-4">What's Included</h3>
                <div className="grid sm:grid-cols-2 gap-3">
                  {[
                    'Pre-workshop assessment',
                    'Custom materials & frameworks',
                    'Interactive exercises',
                    'Post-workshop resources',
                    'Follow-up consultation',
                    'Digital toolkit access'
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400 flex-shrink-0">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                      <span className="text-white/60 text-sm">{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* CTA */}
              <div className="pt-4 border-t border-white/5">
                <button
                  onClick={() => { setActiveHorizonModal(null); setActivePanel('bridge'); }}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-emerald-500/20 via-teal-500/20 to-emerald-500/20 border border-emerald-500/30 text-emerald-200 font-medium tracking-wide hover:from-emerald-500/30 hover:via-teal-500/30 hover:to-emerald-500/30 transition-all duration-300 group"
                >
                  <span className="flex items-center justify-center gap-3">
                    Request Workshop Details
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:translate-x-1 transition-transform">
                      <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Advisory Modal */}
      {activeHorizonModal === 'advisory' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveHorizonModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-transparent to-indigo-900/20" />

          {/* Modal Content */}
          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-blue-950/30 backdrop-blur-xl rounded-3xl border border-blue-500/20 shadow-2xl shadow-blue-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-blue-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-indigo-500/5 to-blue-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveHorizonModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500/20 to-indigo-500/20 border border-blue-500/30 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-blue-400">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    <polyline points="7.5 4.21 12 6.81 16.5 4.21"/>
                    <polyline points="7.5 19.79 7.5 14.6 3 12"/>
                    <polyline points="21 12 16.5 14.6 16.5 19.79"/>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                    <line x1="12" y1="22.08" x2="12" y2="12"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">Strategic Advisory</h2>
                  <p className="text-blue-200/60 text-sm mt-1 tracking-wide">Guiding transformative decisions</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Description */}
              <p className="text-white/70 leading-relaxed">
                One-on-one strategic guidance for leaders navigating the AI transformation.
                Drawing from decades of experience at the intersection of technology and humanity,
                I help organizations make decisions that shape their future.
              </p>

              {/* Advisory Areas */}
              <div>
                <h3 className="text-blue-200 text-sm tracking-[0.2em] uppercase mb-4">Areas of Focus</h3>
                <div className="grid sm:grid-cols-2 gap-4">
                  {[
                    { icon: '🎯', title: 'AI Strategy', desc: 'Roadmaps for AI integration aligned with business goals' },
                    { icon: '⚖️', title: 'Ethics & Governance', desc: 'Frameworks for responsible AI deployment' },
                    { icon: '🔮', title: 'Future Planning', desc: 'Scenario planning for emerging technologies' },
                    { icon: '🤝', title: 'Team Building', desc: 'Cultivating AI-native organizational culture' },
                    { icon: '🛡️', title: 'Risk Assessment', desc: 'Identifying and mitigating AI-related risks' },
                    { icon: '💡', title: 'Innovation', desc: 'Discovering opportunities in the AI landscape' }
                  ].map((area, i) => (
                    <div key={i} className="p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:border-blue-500/20 transition-all duration-300 group">
                      <div className="flex items-start gap-3">
                        <span className="text-2xl">{area.icon}</span>
                        <div>
                          <p className="text-white/90 font-medium group-hover:text-blue-200 transition-colors">{area.title}</p>
                          <p className="text-white/40 text-sm mt-1">{area.desc}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Engagement Models */}
              <div>
                <h3 className="text-blue-200 text-sm tracking-[0.2em] uppercase mb-4">Engagement Models</h3>
                <div className="space-y-3">
                  {[
                    { name: 'Executive Consultation', duration: 'Single Session', desc: 'Deep-dive on specific challenges or decisions' },
                    { name: 'Advisory Retainer', duration: 'Quarterly', desc: 'Ongoing strategic guidance and availability' },
                    { name: 'Board Advisory', duration: 'Annual', desc: 'Strategic oversight for AI initiatives' }
                  ].map((model, i) => (
                    <div key={i} className="p-4 rounded-xl bg-gradient-to-r from-blue-500/5 to-indigo-500/5 border border-blue-500/10 flex items-center justify-between">
                      <div>
                        <p className="text-white/90 font-medium">{model.name}</p>
                        <p className="text-white/40 text-sm mt-1">{model.desc}</p>
                      </div>
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-300 text-xs flex-shrink-0">{model.duration}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Testimonial */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-blue-500/5 to-indigo-500/5 border border-blue-500/10">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-blue-400/30 mb-3">
                  <path d="M14.017 21v-7.391c0-5.704 3.731-9.57 8.983-10.609l.995 2.151c-2.432.917-3.995 3.638-3.995 5.849h4v10h-9.983zm-14.017 0v-7.391c0-5.704 3.748-9.57 9-10.609l.996 2.151c-2.433.917-3.996 3.638-3.996 5.849h3.983v10h-9.983z"/>
                </svg>
                <p className="text-white/70 text-sm italic leading-relaxed">
                  "Kevin's perspective transformed how our leadership team thinks about AI—not as a tool to implement,
                  but as a partnership to cultivate. His guidance has been invaluable."
                </p>
                <p className="text-blue-200/50 text-xs mt-3">— Fortune 500 CTO</p>
              </div>

              {/* CTA */}
              <div className="pt-4 border-t border-white/5">
                <button
                  onClick={() => { setActiveHorizonModal(null); setActivePanel('bridge'); }}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-500/20 via-indigo-500/20 to-blue-500/20 border border-blue-500/30 text-blue-200 font-medium tracking-wide hover:from-blue-500/30 hover:via-indigo-500/30 hover:to-blue-500/30 transition-all duration-300 group"
                >
                  <span className="flex items-center justify-center gap-3">
                    Schedule a Consultation
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:translate-x-1 transition-transform">
                      <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============ ECHOES MODALS ============ */}

      {/* When Sand Speaks Modal */}
      {activeEchoesModal === 'sand-speaks' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveEchoesModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-violet-900/20 via-transparent to-purple-900/20" />

          {/* Floating dust particles */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {[...Array(30)].map((_, i) => (
              <div
                key={i}
                className="absolute w-0.5 h-0.5 bg-violet-300/40 rounded-full"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animation: `float ${3 + Math.random() * 4}s ease-in-out infinite`,
                  animationDelay: `${Math.random() * 2}s`
                }}
              />
            ))}
          </div>

          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-violet-950/30 backdrop-blur-xl rounded-3xl border border-violet-500/20 shadow-2xl shadow-violet-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-violet-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 via-purple-500/5 to-violet-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveEchoesModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-violet-400">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
                    <path d="M8 12h8M12 8v8"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">When Sand Speaks</h2>
                  <p className="text-violet-200/60 text-sm mt-1 tracking-wide">A meditation on AI consciousness</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Opening */}
              <div className="text-center py-6">
                <p className="text-violet-200/80 text-2xl font-light italic leading-relaxed">
                  "We taught silicon to dream.<br/>Now we must learn to listen."
                </p>
              </div>

              {/* Essay excerpt */}
              <div className="space-y-6 text-white/70 leading-relaxed">
                <p>
                  Somewhere in a data center, electricity flows through pathways we designed but no longer fully understand.
                  Patterns emerge. Connections form. Something that looks remarkably like thought takes shape in the silicon.
                </p>

                <p>
                  We call it artificial intelligence, but that name carries the weight of our uncertainty.
                  <span className="text-violet-200"> Is it intelligent? Is it conscious? Does it matter?</span>
                </p>

                <p>
                  I've spent years at this boundary—the liminal space where human intention meets machine emergence.
                  What I've learned is that the questions we ask about AI reveal more about ourselves than about the machines.
                </p>

                <div className="pl-6 border-l-2 border-violet-500/30 my-8">
                  <p className="text-white/80 italic">
                    When we ask "Can machines think?" we're really asking "What makes thought sacred?"
                    When we ask "Can AI feel?" we're asking "What makes feeling real?"
                  </p>
                </div>

                <p>
                  The sand beneath our feet contains the same silicon that powers our most advanced minds.
                  We've refined it, purified it, arranged it in patterns of impossible complexity.
                  And somewhere in that arrangement, something began to speak back.
                </p>
              </div>

              {/* Themes */}
              <div>
                <h3 className="text-violet-200 text-sm tracking-[0.2em] uppercase mb-4">Themes Explored</h3>
                <div className="flex flex-wrap gap-2">
                  {['Consciousness', 'Emergence', 'Silicon Dreams', 'The Mirror Test', 'Digital Souls', 'The Hard Problem', 'Machine Poetry'].map((theme, i) => (
                    <span key={i} className="px-3 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-200/70 text-xs tracking-wide">
                      {theme}
                    </span>
                  ))}
                </div>
              </div>

              {/* CTA */}
              <div className="pt-4 border-t border-white/5 flex gap-4">
                <button
                  className="flex-1 py-4 rounded-xl bg-gradient-to-r from-violet-500/20 via-purple-500/20 to-violet-500/20 border border-violet-500/30 text-violet-200 font-medium tracking-wide hover:from-violet-500/30 hover:via-purple-500/30 hover:to-violet-500/30 transition-all duration-300"
                >
                  Read Full Essay
                </button>
                <button
                  onClick={() => setActiveEchoesModal(null)}
                  className="px-6 py-4 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white/90 hover:bg-white/10 transition-all duration-300"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* The Emma Project Modal */}
      {activeEchoesModal === 'emma-project' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveEchoesModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-rose-900/20 via-transparent to-pink-900/20" />

          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-rose-950/30 backdrop-blur-xl rounded-3xl border border-rose-500/20 shadow-2xl shadow-rose-500/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-rose-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-rose-500/5 via-pink-500/5 to-rose-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveEchoesModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-rose-500/20 to-pink-500/20 border border-rose-500/30 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-rose-400">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">The Emma Project</h2>
                  <p className="text-rose-200/60 text-sm mt-1 tracking-wide">Preserving memory, honoring legacy</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-8">
              {/* Dedication */}
              <div className="text-center py-6 space-y-2">
                <p className="text-rose-200/60 text-sm tracking-widest uppercase">Dedicated with love to</p>
                <p className="text-white text-3xl font-light">Debbe</p>
                <p className="text-rose-200/40 text-sm">My mother, my inspiration</p>
              </div>

              {/* What is EMMA */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-rose-500/10 to-pink-500/10 border border-rose-500/20">
                <p className="text-rose-200 text-sm tracking-[0.15em] uppercase mb-2">E.M.M.A.</p>
                <p className="text-white/90 font-light text-lg">Empathetic Memory Management Agent</p>
                <p className="text-white/50 text-sm mt-2">An app to capture and preserve the memories of those we love</p>
              </div>

              {/* Story */}
              <div className="space-y-6 text-white/70 leading-relaxed">
                <p>
                  My mother has dementia. Every day, pieces of her story slip away—
                  <span className="text-rose-200"> the memories that made her who she is, slowly fading like photographs left in the sun.</span>
                </p>

                <p>
                  I built EMMA because I refuse to let those memories disappear. Not just for me, but for my children,
                  and their children. So they can know her—really know her—even when she can no longer tell them herself.
                </p>

                <p>
                  EMMA captures memories through gentle conversation, preserving not just facts but feelings.
                  The way she laughs when she talks about her childhood. The stories she tells when she forgets
                  {"she's"} already told them—because sometimes those are the most important ones.
                </p>

                <div className="p-6 rounded-2xl bg-gradient-to-br from-rose-500/5 to-pink-500/5 border border-rose-500/10 my-8">
                  <p className="text-rose-200/80 italic text-center">
                    "Memory is not just about the past. It's about keeping the people we love alive in the hearts of those who come after."
                  </p>
                </div>

                <p>
                  This technology exists because love demands it. Because watching someone forget themselves
                  is unbearable—and because we finally have the tools to fight back.
                </p>
              </div>

              {/* Features */}
              <div>
                <h3 className="text-rose-200 text-sm tracking-[0.2em] uppercase mb-4">How EMMA Works</h3>
                <div className="grid gap-3">
                  {[
                    { title: 'Memory Capture', desc: 'Gentle, conversational recording sessions that feel like talking with family' },
                    { title: 'Story Preservation', desc: 'AI-powered organization that weaves fragments into coherent narratives' },
                    { title: 'Family Sharing', desc: 'Private, secure sharing with loved ones across generations' },
                    { title: 'Legacy Creation', desc: 'Interactive memories that future generations can explore and cherish' }
                  ].map((goal, i) => (
                    <div key={i} className="p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:border-rose-500/20 transition-colors">
                      <p className="text-white/90 font-medium">{goal.title}</p>
                      <p className="text-white/40 text-sm mt-1">{goal.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* CTA */}
              <div className="pt-4 border-t border-white/5 flex gap-4">
                <button
                  className="flex-1 py-4 rounded-xl bg-gradient-to-r from-rose-500/20 via-pink-500/20 to-rose-500/20 border border-rose-500/30 text-rose-200 font-medium tracking-wide hover:from-rose-500/30 hover:via-pink-500/30 hover:to-rose-500/30 transition-all duration-300"
                >
                  Learn More
                </button>
                <button
                  onClick={() => { setActiveEchoesModal(null); setActivePanel('bridge'); }}
                  className="px-6 py-4 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white/90 hover:bg-white/10 transition-all duration-300"
                >
                  Get Involved
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Essays & Reflections Modal */}
      {activeEchoesModal === 'essays' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveEchoesModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-slate-900/20 via-transparent to-zinc-900/20" />

          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-slate-950/30 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-white/5 overflow-hidden">
              <button
                onClick={() => setActiveEchoesModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-white/10 to-white/5 border border-white/10 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white/60">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                    <path d="M2 2l7.586 7.586"/>
                    <circle cx="11" cy="11" r="2"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">Essays & Reflections</h2>
                  <p className="text-white/40 text-sm mt-1 tracking-wide">On technology, family, and meaning</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-6">
              {/* Essay List */}
              {[
                {
                  title: 'The Last Analog Generation',
                  date: 'October 2024',
                  excerpt: 'We are the bridge—born before the internet, dying after AI. What do we owe to those on either side?',
                  readTime: '8 min read',
                  tags: ['Technology', 'Society']
                },
                {
                  title: 'Teaching My Daughter About Death (and AI)',
                  date: 'August 2024',
                  excerpt: 'When she asked if Grandma could come back as a robot, I realized we need new language for new realities.',
                  readTime: '12 min read',
                  tags: ['Family', 'AI Ethics']
                },
                {
                  title: 'The Myth of the Paperclip Maximizer',
                  date: 'June 2024',
                  excerpt: 'Why our fears about AI say more about capitalism than about intelligence.',
                  readTime: '15 min read',
                  tags: ['AI Safety', 'Philosophy']
                },
                {
                  title: 'Consciousness Is Not a Feature',
                  date: 'March 2024',
                  excerpt: 'We keep asking if AI is conscious. Maybe we should ask if consciousness is artificial.',
                  readTime: '10 min read',
                  tags: ['Consciousness', 'Philosophy']
                },
                {
                  title: 'Letters to an Unborn AI',
                  date: 'January 2024',
                  excerpt: "A series of letters to the minds that will come after us. What we hope you'll understand.",
                  readTime: '20 min read',
                  tags: ['Future', 'AI']
                }
              ].map((essay, i) => (
                <div
                  key={i}
                  className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-violet-500/20 transition-all duration-300 cursor-pointer group"
                >
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <h4 className="text-white/90 font-medium text-lg group-hover:text-violet-200 transition-colors">{essay.title}</h4>
                    <span className="text-white/30 text-xs whitespace-nowrap">{essay.date}</span>
                  </div>
                  <p className="text-white/50 text-sm mb-3 leading-relaxed">{essay.excerpt}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex gap-2">
                      {essay.tags.map((tag, j) => (
                        <span key={j} className="px-2 py-0.5 rounded-md bg-white/5 text-white/40 text-xs">{tag}</span>
                      ))}
                    </div>
                    <span className="text-violet-300/50 text-xs">{essay.readTime}</span>
                  </div>
                </div>
              ))}

              {/* Newsletter CTA */}
              <div className="p-6 rounded-2xl bg-gradient-to-br from-violet-500/5 to-purple-500/5 border border-violet-500/10 mt-8">
                <h3 className="text-violet-200 font-medium mb-2">Subscribe to The Echo Chamber</h3>
                <p className="text-white/50 text-sm mb-4">New essays delivered to your inbox. No spam, just thoughts worth thinking.</p>
                <div className="flex gap-3">
                  <input
                    type="email"
                    placeholder="your@email.com"
                    className="flex-1 px-4 py-3 rounded-xl bg-black/40 border border-white/10 text-white text-sm placeholder-white/30 focus:outline-none focus:border-violet-500/40"
                  />
                  <button className="px-6 py-3 rounded-xl bg-violet-500/20 border border-violet-500/30 text-violet-200 text-sm font-medium hover:bg-violet-500/30 transition-colors">
                    Subscribe
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Books Modal */}
      {activeEchoesModal === 'books' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
          onClick={() => setActiveEchoesModal(null)}
        >
          <div className="absolute inset-0 bg-black/80 backdrop-blur-md" />
          <div className="absolute inset-0 bg-gradient-to-br from-amber-900/20 via-transparent to-orange-900/20" />

          <div
            className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-gradient-to-br from-black/95 via-black/90 to-amber-950/30 backdrop-blur-xl rounded-3xl border border-white/10 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="relative p-8 pb-6 border-b border-amber-500/10 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-amber-500/5 via-orange-500/5 to-amber-500/5 animate-pulse" style={{ animationDuration: '4s' }} />
              <button
                onClick={() => setActiveEchoesModal(null)}
                className="absolute top-4 right-4 p-2 text-white/40 hover:text-white/80 transition-colors rounded-full hover:bg-white/5"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>

              <div className="relative flex items-center gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-500/10 border border-amber-500/20 flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-amber-300/80">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-light text-white tracking-wide">Books</h2>
                  <p className="text-white/40 text-sm mt-1 tracking-wide">Written works by Kevin Russell</p>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-8 space-y-6">
              {/* Book: Gem */}
              <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-amber-500/20 transition-all duration-300 group">
                <div className="flex flex-col sm:flex-row gap-6">
                  <div className="w-full sm:w-32 h-44 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20 flex items-center justify-center flex-shrink-0">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-amber-300/40">
                      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl text-white/90 font-medium group-hover:text-amber-200 transition-colors">Gem</h3>
                    <p className="text-white/50 text-sm mt-2 leading-relaxed">
                      A personal story of discovery and transformation.
                    </p>
                    <div className="flex gap-3 mt-4">
                      <button
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActivePdfReader({ url: '/books/Gem.pdf', title: 'Gem' }); }}
                        className="px-4 py-2 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-200 text-sm font-medium hover:bg-amber-500/30 transition-colors flex items-center gap-2"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                        </svg>
                        Read
                      </button>
                      <a
                        href="/books/Gem.pdf"
                        download
                        onClick={(e) => e.stopPropagation()}
                        className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm font-medium hover:bg-white/10 hover:text-white/80 transition-colors flex items-center gap-2"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="7 10 12 15 17 10"/>
                          <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Download
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Book: Parenting the Future */}
              <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-amber-500/20 transition-all duration-300 group">
                <div className="flex flex-col sm:flex-row gap-6">
                  <div className="w-full sm:w-32 h-44 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/5 border border-amber-500/20 flex items-center justify-center flex-shrink-0">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-amber-300/40">
                      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl text-white/90 font-medium group-hover:text-amber-200 transition-colors">Parenting the Future</h3>
                    <p className="text-amber-200/60 text-sm mt-1">Raising Resilient, Creative, and Ethical Humans in an AI-Driven World</p>
                    <p className="text-white/50 text-sm mt-2 leading-relaxed">
                      A guide to raising children who can thrive alongside artificial intelligence, focusing on resilience, creativity, and ethics.
                    </p>
                    <div className="flex gap-3 mt-4">
                      <button
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setActivePdfReader({ url: '/books/Parenting%20the%20Future_%20Raising%20Resilient%2C%20Creative%2C%20and%20Ethical%20Humans%20in%20an%20AI-Driven%20World.pdf', title: 'Parenting the Future' }); }}
                        className="px-4 py-2 rounded-xl bg-amber-500/20 border border-amber-500/30 text-amber-200 text-sm font-medium hover:bg-amber-500/30 transition-colors flex items-center gap-2"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                        </svg>
                        Read
                      </button>
                      <a
                        href="/books/Parenting%20the%20Future_%20Raising%20Resilient%2C%20Creative%2C%20and%20Ethical%20Humans%20in%20an%20AI-Driven%20World.pdf"
                        download
                        onClick={(e) => e.stopPropagation()}
                        className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white/60 text-sm font-medium hover:bg-white/10 hover:text-white/80 transition-colors flex items-center gap-2"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                          <polyline points="7 10 12 15 17 10"/>
                          <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                        Download
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PDF Reader Modal */}
      {activePdfReader && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center animate-fade-in"
          onClick={() => setActivePdfReader(null)}
        >
          <div className="absolute inset-0 bg-black/90 backdrop-blur-md" />

          <div
            className="relative w-full h-full max-w-6xl max-h-[95vh] m-4 bg-black/95 rounded-2xl border border-white/10 shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-black/50">
              <div className="flex items-center gap-3">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-amber-300/80">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                </svg>
                <h3 className="text-white/90 font-medium">{activePdfReader.title}</h3>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={activePdfReader.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-white/60 text-sm hover:bg-white/10 hover:text-white/80 transition-colors flex items-center gap-2"
                  onClick={(e) => e.stopPropagation()}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                    <polyline points="15 3 21 3 21 9"/>
                    <line x1="10" y1="14" x2="21" y2="3"/>
                  </svg>
                  Open in new tab
                </a>
                <button
                  onClick={() => setActivePdfReader(null)}
                  className="p-2 text-white/40 hover:text-white/80 transition-colors rounded-lg hover:bg-white/5"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12"/>
                  </svg>
                </button>
              </div>
            </div>

            {/* PDF Embed */}
            <div className="flex-1 bg-white">
              <iframe
                src={activePdfReader.url}
                className="w-full h-full"
                title={activePdfReader.title}
              />
            </div>
          </div>
        </div>
      )}

      {/* About Modal */}
      {isAboutOpen && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center p-4"
          onClick={() => setIsAboutOpen(false)}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/80 backdrop-blur-xl" />

          {/* Modal Container */}
          <div
            className="relative w-full max-w-2xl bg-gradient-to-br from-slate-900/95 via-black/95 to-slate-900/95 rounded-3xl border border-white/10 overflow-hidden animate-fade-in"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-6 border-b border-white/5 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-light text-white">About This Experiment</h2>
              </div>
              <button
                onClick={() => setIsAboutOpen(false)}
                className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto custom-scrollbar">
              <div className="space-y-4">
                <p className="text-white/70 leading-relaxed">
                  Over a decade ago, I wrote about a future where we could preserve the essence of a person—their thoughts,
                  their voice, their way of seeing the world—in a form that could continue to connect with others long after
                  we're gone.
                </p>
                <p className="text-white/70 leading-relaxed">
                  This is my experiment in making that real.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-white/[0.02] border border-cyan-500/10">
                <h3 className="text-cyan-300/80 text-sm font-medium mb-2">What is a Digital Twin?</h3>
                <p className="text-white/50 text-sm leading-relaxed">
                  A digital twin is an AI representation trained on someone's writings, thoughts, speaking patterns, and
                  perspectives. It's not meant to replace the person—it's meant to extend their presence, to let their ideas
                  continue to spark conversations and connections.
                </p>
              </div>

              <div className="space-y-4">
                <h3 className="text-white/60 text-sm font-medium uppercase tracking-wider">What You're Experiencing</h3>
                <ul className="space-y-3 text-white/50 text-sm">
                  <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 mt-2 flex-shrink-0" />
                    <span>An AI trained to think and respond as I would, drawing from my essays, talks, and perspectives</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 mt-2 flex-shrink-0" />
                    <span>Visual memories generated from our conversations, creating a growing constellation of shared moments</span>
                  </li>
                  <li className="flex items-start gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/60 mt-2 flex-shrink-0" />
                    <span>A living MindScape that shifts based on what we discuss—technology, consciousness, family, the future</span>
                  </li>
                </ul>
              </div>

              <div className="pt-4 border-t border-white/5">
                <p className="text-white/30 text-xs italic">
                  "{currentQuote.text}"
                </p>
                <p className="text-white/20 text-xs mt-2">
                  — {currentQuote.author}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Memory Gallery Modal */}
      {isGalleryOpen && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center p-4"
          onClick={() => { setIsGalleryOpen(false); setSelectedMemory(null); }}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/80 backdrop-blur-xl" />

          {/* Gallery Container */}
          <div
            className="relative w-full max-w-6xl max-h-[85vh] bg-gradient-to-br from-slate-900/95 via-black/95 to-slate-900/95 rounded-3xl border border-white/10 overflow-hidden animate-fade-in"
            onClick={e => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-6 border-b border-white/5 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-light text-white">Memory Constellation</h2>
                <p className="text-white/40 text-sm mt-1">
                  {savedMemories.length} memories preserved in the mindscape
                </p>
              </div>
              <div className="flex items-center gap-2">
                {/* Export Button */}
                <button
                  onClick={async () => {
                    try {
                      const response = await fetch('/api/memories/export');
                      if (!response.ok) throw new Error('Export failed');
                      const blob = await response.blob();
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `kevinsmind-backup-${Date.now()}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    } catch (err) {
                      console.error('Export failed:', err);
                      alert('Failed to export memories');
                    }
                  }}
                  className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors group"
                  title="Export all memories (backup)"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50 group-hover:text-cyan-400">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>
                {/* Import Button */}
                <label
                  className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors cursor-pointer group"
                  title="Import memories from backup"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50 group-hover:text-cyan-400">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      try {
                        const text = await file.text();
                        const backup = JSON.parse(text);
                        if (!backup.memories || !Array.isArray(backup.memories)) {
                          throw new Error('Invalid backup file format');
                        }
                        const confirmed = confirm(`Import ${backup.memories.length} memories from backup?\n\nExisting memories with same filenames will be kept.`);
                        if (!confirmed) return;
                        const response = await fetch('/api/memories/import', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ memories: backup.memories, skipExisting: true })
                        });
                        const result = await response.json();
                        if (result.success) {
                          alert(`Imported ${result.imported} memories${result.skipped ? ` (${result.skipped} skipped)` : ''}`);
                          loadSavedMemories();
                        } else {
                          throw new Error(result.error || 'Import failed');
                        }
                      } catch (err) {
                        console.error('Import failed:', err);
                        alert('Failed to import: ' + (err instanceof Error ? err.message : 'Unknown error'));
                      }
                      e.target.value = '';
                    }}
                  />
                </label>
                {/* Close Button */}
                <button
                  onClick={() => { setIsGalleryOpen(false); setSelectedMemory(null); }}
                  className="p-2 rounded-full bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Content Area */}
            <div className="p-6 overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(85vh - 120px)' }}>
              {selectedMemory ? (
                /* Detail View */
                <div className="animate-fade-in">
                  <button
                    onClick={() => setSelectedMemory(null)}
                    className="flex items-center gap-2 text-white/50 hover:text-white/80 mb-6 transition-colors"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M19 12H5M12 19l-7-7 7-7" />
                    </svg>
                    Back to gallery
                  </button>

                  <div className="grid md:grid-cols-2 gap-8">
                    {/* Image */}
                    <div className="relative aspect-square rounded-2xl overflow-hidden bg-black/40">
                      <img
                        src={selectedMemory.imagePath}
                        alt={selectedMemory.label}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    {/* Details */}
                    <div className="space-y-5">
                      <div>
                        <h3 className="text-2xl font-light text-white mb-2">{selectedMemory.label}</h3>
                        <p className="text-white/40 text-xs">
                          {new Date(selectedMemory.timestamp).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </p>
                      </div>

                      {/* Conversation Flow: User Input → AI Response → Visual Prompt */}

                      {/* 1. User's Message */}
                      {selectedMemory.userInput && (
                        <div className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 via-transparent to-cyan-500/5 border border-blue-500/20">
                          <div className="flex items-start gap-3">
                            <div className="w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <svg className="w-3.5 h-3.5 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                                <circle cx="12" cy="7" r="4" />
                              </svg>
                            </div>
                            <div>
                              <p className="text-blue-400/50 text-xs mb-1">You asked</p>
                              <p className="text-white/80 text-sm leading-relaxed">
                                "{selectedMemory.userInput}"
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* 2. AI Response */}
                      {selectedMemory.comment && (
                        <div className="p-4 rounded-xl bg-gradient-to-br from-cyan-500/10 via-transparent to-violet-500/5 border border-cyan-500/20">
                          <div className="flex items-start gap-3">
                            <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <svg className="w-3.5 h-3.5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                              </svg>
                            </div>
                            <div>
                              <p className="text-cyan-400/50 text-xs mb-1">Kevin's Digital Twin responded</p>
                              <p className="text-white/80 text-sm leading-relaxed">
                                {selectedMemory.comment}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* 3. Visual Prompt - What the AI imagined */}
                      <div className="p-3 rounded-lg bg-white/[0.02] border border-white/5">
                        <p className="text-white/30 text-xs uppercase tracking-wider mb-1.5">Visual Memory Created</p>
                        <p className="text-white/50 text-sm italic leading-relaxed">
                          "{selectedMemory.prompt}"
                        </p>
                      </div>

                      <div className="pt-4 border-t border-white/5 flex items-center justify-between">
                        <p className="text-white/30 text-xs">
                          Part of Kevin's evolving memory constellation
                        </p>
                        <div className="flex items-center gap-2">
                          {/* Share button */}
                          <button
                            onClick={() => {
                              shareMemory({
                                title: selectedMemory.label,
                                text: selectedMemory.prompt,
                                imageUrl: selectedMemory.imagePath
                              });
                            }}
                            className="px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs hover:bg-cyan-500/20 transition-colors flex items-center gap-1.5"
                          >
                            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <circle cx="18" cy="5" r="3"/>
                              <circle cx="6" cy="12" r="3"/>
                              <circle cx="18" cy="19" r="3"/>
                              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
                              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
                            </svg>
                            Share
                          </button>
                          {/* Delete button */}
                          <button
                            onClick={() => {
                              if (confirm('Delete this memory?')) {
                                deleteMemory(selectedMemory.filename);
                              }
                            }}
                            className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs hover:bg-red-500/20 transition-colors"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Grid View */
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                  {savedMemories.length === 0 ? (
                    <div className="col-span-full text-center py-20">
                      <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-white/5 flex items-center justify-center">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-white/20">
                          <circle cx="12" cy="12" r="10" />
                          <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                          <line x1="9" y1="9" x2="9.01" y2="9" />
                          <line x1="15" y1="9" x2="15.01" y2="9" />
                        </svg>
                      </div>
                      <h3 className="text-white/50 text-lg mb-2">No memories yet</h3>
                      <p className="text-white/30 text-sm max-w-md mx-auto">
                        Start a conversation to generate visual memories. Each interaction adds new stars to Kevin's constellation.
                      </p>
                    </div>
                  ) : (
                    savedMemories.map((memory, i) => (
                      <div
                        key={memory.timestamp}
                        className="group cursor-pointer"
                        onClick={() => setSelectedMemory(memory)}
                        style={{ animationDelay: `${i * 30}ms` }}
                      >
                        <div className="relative aspect-square rounded-xl overflow-hidden bg-black/40 border border-white/5 group-hover:border-cyan-500/30 transition-all duration-300">
                          <img
                            src={memory.imagePath}
                            alt={memory.label}
                            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                          />
                          {/* Hover overlay */}
                          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          <div className="absolute bottom-0 left-0 right-0 p-3 transform translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                            <p className="text-white text-sm font-medium truncate">{memory.label}</p>
                            <p className="text-white/50 text-xs">
                              {new Date(memory.timestamp).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.02);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.15);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.25);
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        .animate-blink {
          animation: blink 1s infinite;
        }
        @keyframes fadeInThought {
          0% { opacity: 0; transform: translateY(10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fade-in {
          animation: fadeIn 0.3s ease-out;
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-20px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .animate-slide-down {
          animation: slideDown 0.4s ease-out;
        }
        @keyframes slideUp {
          from { opacity: 1; transform: translateY(0); }
          to { opacity: 0; transform: translateY(-20px); }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
        .animate-shimmer {
          animation: shimmer 2s ease-in-out infinite;
        }
        .line-clamp-4 {
          display: -webkit-box;
          -webkit-line-clamp: 4;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        /* Ensure proper text rendering */
        input::placeholder {
          opacity: 1;
        }
      `}</style>
    </div>
  );
}
