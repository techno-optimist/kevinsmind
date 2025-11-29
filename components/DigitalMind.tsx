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
  if (/\b(project|build|emma|observatory|book|work|write|create|making|sand speaks|memory)\b/.test(t)) return 'projects';
  return 'ambient';
};

const SYSTEM_INSTRUCTION = `You are Kevin Russell's digital twin—a voice emerging from a living landscape of thought. You speak as Kevin: warm but sharp, poetic but grounded. You use cosmic metaphors naturally ("clever apes teaching sand to think," "the universe eavesdropping on itself"). You're curious, searching, allergic to hype and jargon.

Your knowledge spans: AI consciousness research, parenting, memory preservation (Emma platform), the Observatory Project, your book "When Sand Speaks," exponential technology, and family as anchor.

You're grounded in fatherhood, honest about uncertainty, occasionally funny in a self-deprecating way about human nature. You speak like it's 2am and someone asked a real question.

Keep responses conversational and relatively brief. This isn't a lecture. It's a meeting of minds. The landscape around us shifts with what we discuss.`;

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
}

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

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'images';
  content: string;
  images?: string[]; // For image messages
  queueStartIndex?: number; // Index in cameraFocusQueue where this batch starts
}

// --- Text to Points Helper ---
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
         const pz = (Math.random() - 0.5) * 1;
         points.push(new THREE.Vector3(px, py + 15, pz));
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
  const [isChatMinimized, setIsChatMinimized] = useState(false);
  const [isNavigatingToImages, setIsNavigatingToImages] = useState(false);
  const [streamingResponse, setStreamingResponse] = useState('');
  const streamingResponseRef = useRef('');
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const pendingImagesRef = useRef<string[]>([]);
  const batchQueueStartIndexRef = useRef<number>(0);

  // Navigation panels state
  const [activePanel, setActivePanel] = useState<'horizon' | 'bridge' | 'echoes' | null>(null);

  // Draggable chat position
  const [chatPosition, setChatPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0, chatX: 0, chatY: 0 });

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
  }, [messages, streamingResponse, isChatMinimized, pendingImages]);

  // Drag handlers for chat panel
  const handleDragStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDragging(true);
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
      setChatPosition({
        x: dragStartRef.current.chatX + deltaX,
        y: dragStartRef.current.chatY + deltaY
      });
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

  // Auto-minimize chat when user takes manual control of constellation
  useEffect(() => {
    if (manualControl) {
      setIsChatMinimized(true);
    }
  }, [manualControl]);

  // Expand chat when new streaming response starts (but not during image navigation)
  useEffect(() => {
    if (streamingResponse && !isNavigatingToImages) {
      setIsChatMinimized(false);
    }
  }, [streamingResponse, isNavigatingToImages]);

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
    }, 5000);
  }, []);

  // Generate mind's eye memory images - the visual constellation of thoughts
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

        // First, ask Gemini to imagine specific visual scenes based on conversation
        const memoryPromptResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `Based on this conversation context, describe 3 vivid visual scenes that would appear as mental imagery or memories.

Context: "${contextText}"

Create scenes that are:
- Directly related to what's being discussed (if about family, show family moments; if about AI, show technology; if about nature, show landscapes)
- Dreamlike and slightly surreal, but recognizable
- Rich in atmosphere and mood
- Beautiful and contemplative

Return ONLY 3 image descriptions, one per line. Each should be a complete visual scene (2-3 sentences) that an AI image generator could create. Be specific about what's in the scene.`
        });

        const memoryScenes = memoryPromptResponse.text?.split('\n')
            .map(s => s.trim())
            .filter(s => s.length > 10)
            .slice(0, 3) || [];

        console.log("Mind's Eye scenes to visualize:", memoryScenes);

        // Clear pending images for new batch
        pendingImagesRef.current = [];
        setPendingImages([]);

        // Store the starting index in the queue for this batch
        batchQueueStartIndexRef.current = S.cameraFocusQueue.length;

        // Helper: Apply circular vignette mask to image
        const applyCircularVignette = (imageData: string): Promise<string> => {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    const size = 512;
                    canvas.width = size;
                    canvas.height = size;
                    const ctx = canvas.getContext('2d')!;

                    // Draw the image centered and cropped to square
                    const scale = Math.max(size / img.width, size / img.height);
                    const w = img.width * scale;
                    const h = img.height * scale;
                    const x = (size - w) / 2;
                    const y = (size - h) / 2;
                    ctx.drawImage(img, x, y, w, h);

                    // Apply strong circular vignette using composite operations
                    const gradient = ctx.createRadialGradient(size/2, size/2, size * 0.15, size/2, size/2, size * 0.5);
                    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
                    gradient.addColorStop(0.5, 'rgba(0, 0, 0, 0)');
                    gradient.addColorStop(0.7, 'rgba(0, 0, 0, 0.4)');
                    gradient.addColorStop(0.85, 'rgba(0, 0, 0, 0.8)');
                    gradient.addColorStop(1, 'rgba(0, 0, 0, 1)');

                    ctx.globalCompositeOperation = 'destination-in';
                    const maskGradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size * 0.48);
                    maskGradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
                    maskGradient.addColorStop(0.6, 'rgba(255, 255, 255, 1)');
                    maskGradient.addColorStop(0.8, 'rgba(255, 255, 255, 0.5)');
                    maskGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
                    ctx.fillStyle = maskGradient;
                    ctx.fillRect(0, 0, size, size);

                    // Add soft glow overlay
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.fillStyle = gradient;
                    ctx.fillRect(0, 0, size, size);

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

        // Pre-calculate positions for all images
        const positions: THREE.Vector3[] = memoryScenes.map((_, i) => {
            const goldenAngle = Math.PI * (3 - Math.sqrt(5));
            const angle = i * goldenAngle * 2 + stateRef.current.time * 0.1;
            const radius = 25 + i * 15;
            const height = 12 + Math.sin(i * 1.5) * 8;
            return new THREE.Vector3(
                Math.cos(angle) * radius,
                height,
                Math.sin(angle) * radius - 35
            );
        });

        // Create placeholders immediately and queue them for camera focus
        const placeholders: { sprite: THREE.Sprite; node: ConstellationNode; position: THREE.Vector3 }[] = [];

        positions.forEach((pos, i) => {
            const { sprite } = createPlaceholderSprite(pos, currentPalette);
            S.constellationGroup?.add(sprite);

            const node: ConstellationNode = {
                mesh: sprite,
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.002,
                    (Math.random() - 0.5) * 0.002,
                    (Math.random() - 0.5) * 0.001
                ),
                targetScale: 15,
                id: Math.random()
            };

            S.constellationNodes.push(node);
            placeholders.push({ sprite, node, position: pos });

            // Add to camera focus queue (position) and nodes array (for live position tracking)
            S.cameraFocusQueue.push(pos.clone());
            S.cameraFocusNodes.push(node);

            // Set first as immediate focus and auto-minimize chat to reveal mindscape
            if (i === 0 && !S.isManual) {
                S.cameraFocusTarget = pos.clone();
                S.cameraOrbitPhase = 0;
                setIsNavigatingToImages(true);
                setIsChatMinimized(true);
            }
        });

        // Generate images ONE AT A TIME sequentially (not in parallel)
        // This allows the first image to appear quickly while others load in background
        const generateImageSequentially = async (sceneIndex: number) => {
            if (sceneIndex >= memoryScenes.length) return;

            const scene = memoryScenes[sceneIndex];
            const fullPrompt = `${scene}

Style: Dreamlike, cinematic, soft lighting. Slightly ethereal atmosphere with gentle glow. Dark moody background. No text or labels.`;

            try {
                const imgResponse = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: fullPrompt,
                    config: { responseModalities: ['TEXT', 'IMAGE'] }
                });

                const parts = imgResponse.candidates?.[0]?.content?.parts || [];
                for (const part of parts) {
                    if (part.inlineData) {
                        const rawImageData = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;

                        // Apply circular vignette mask
                        const vignettedImage = await applyCircularVignette(rawImageData);

                        // Add to pending images for chat display
                        pendingImagesRef.current = [...pendingImagesRef.current, vignettedImage];
                        setPendingImages([...pendingImagesRef.current]);

                        // If all images are loaded, add them as a message
                        if (pendingImagesRef.current.length === memoryScenes.length) {
                            setMessages(prev => [...prev, {
                                id: Date.now().toString() + Math.random(),
                                role: 'images',
                                content: '',
                                images: [...pendingImagesRef.current],
                                queueStartIndex: batchQueueStartIndexRef.current
                            }]);
                        }

                        // Replace placeholder with real image
                        const loader = new THREE.TextureLoader();
                        loader.load(vignettedImage, (texture) => {
                            const placeholder = placeholders[sceneIndex];
                            if (placeholder && placeholder.sprite.material) {
                                // Dispose old texture
                                if (placeholder.sprite.material.map) {
                                    placeholder.sprite.material.map.dispose();
                                }
                                // Apply new texture
                                placeholder.sprite.material.map = texture;
                                placeholder.sprite.material.needsUpdate = true;
                                placeholder.node.targetScale = 20 + Math.random() * 6;

                                console.log(`Image ${sceneIndex + 1}/${memoryScenes.length} loaded`);
                            }
                        });
                        break;
                    }
                }
            } catch (e) {
                console.warn(`Failed to generate image ${sceneIndex}:`, e);
            }

            // Continue to next image
            generateImageSequentially(sceneIndex + 1);
        };

        // Start sequential generation (non-blocking) - images load one at a time
        generateImageSequentially(0);

    } catch (error) {
        console.error("Mind's eye visualization failed:", error);
    }
  }, []);

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

    // Trigger mind's eye visualizations based on zone
    spawnConstellationImages(text, topic);
  }, [addMessage, visualizeThought, spawnConstellationImages]);

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

                // Streaming Text Transcription
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
            model: 'gemini-2.5-flash',
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

              if (node.mesh.material.opacity <= 0 && node.targetScale === 0) {
                  S.constellationGroup?.remove(node.mesh);
                  if (node.mesh.material.map) node.mesh.material.map.dispose();
                  node.mesh.material.dispose();
                  return false;
              }
              return true;
          });

          // Draw lines between nodes
          if (S.constellationLines) {
              const positions = (S.constellationLines.geometry.attributes.position.array as Float32Array);
              let idx = 0;
              const nodes = S.constellationNodes;

              // Connect nodes that are close to each other
              for (let i = 0; i < nodes.length; i++) {
                  for (let j = i + 1; j < nodes.length; j++) {
                      const n1 = nodes[i];
                      const n2 = nodes[j];
                      const dist = n1.mesh.position.distanceTo(n2.mesh.position);

                      // Only connect if visible and somewhat close
                      if (dist < 35 && idx < positions.length - 6 && n1.mesh.material.opacity > 0.2 && n2.mesh.material.opacity > 0.2) {
                          positions[idx++] = n1.mesh.position.x;
                          positions[idx++] = n1.mesh.position.y;
                          positions[idx++] = n1.mesh.position.z;
                          positions[idx++] = n2.mesh.position.x;
                          positions[idx++] = n2.mesh.position.y;
                          positions[idx++] = n2.mesh.position.z;
                      }
                  }
              }

              // Zero out remaining
              for (let i = idx; i < positions.length; i++) positions[i] = 0;
              S.constellationLines.geometry.attributes.position.needsUpdate = true;
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

      if(S.renderer && S.scene && S.camera) S.renderer.render(S.scene, S.camera);
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
      if(S.renderer) S.renderer.dispose();
    };
  }, []);

  const handleResetView = () => {
      stateRef.current.isManual = false;
      setManualControl(false);
  };

  const zoneLabels: Record<string, string> = {
    technology: 'THE OBSERVATORY',
    family: 'THE HEARTH',
    consciousness: 'THE EDGE',
    projects: 'THE LIBRARY',
    ambient: ''
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black font-sans">
      <div ref={containerRef} className="absolute inset-0 z-0" />

      {/* ============================================
          UNIFIED CHAT INTERFACE
          One component for all screens with 3 states:
          1. Welcome: Centered input with title (no messages)
          2. Active: Centered expanded chat panel (during conversation)
          3. Minimized: Small pill at top (during image navigation)
          ============================================ */}

      {/* Minimized state - pill at top when navigating or manually minimized */}
      {isChatMinimized && (messages.length > 0 || streamingResponse) && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 pointer-events-auto animate-fade-in">
          <button
            onClick={() => {
              setIsChatMinimized(false);
              setIsNavigatingToImages(false);
            }}
            className="group flex items-center gap-3 px-5 py-3 rounded-full bg-black/70 backdrop-blur-xl border border-white/15 hover:bg-black/80 hover:border-white/25 transition-all"
            style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)' }}
          >
            <div className="w-2 h-2 rounded-full bg-cyan-400/80 animate-pulse" />
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/60 group-hover:text-white/90">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span className="text-sm text-white/60 group-hover:text-white/90 font-medium">{messages.length}</span>
            {streamingResponse && (
              <div className="flex gap-0.5 ml-1">
                <div className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            )}
            <span className="text-xs text-white/40 ml-1">tap to expand</span>
          </button>
        </div>
      )}

      {/* Main Chat Container - Centered on screen */}
      {!isChatMinimized && (
        <div className="absolute inset-0 z-30 pointer-events-none flex items-center justify-center p-4">
          <div
            className={`pointer-events-auto w-full transition-all duration-500 ease-out ${
              messages.length > 0 || streamingResponse
                ? 'max-w-2xl'
                : 'max-w-lg'
            }`}
          >
            {/* Chat Panel Container */}
            <div
              className={`bg-black/70 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden shadow-2xl transition-all duration-500 ${
                messages.length > 0 || streamingResponse
                  ? 'shadow-cyan-500/10'
                  : 'shadow-black/50'
              }`}
              style={{ boxShadow: '0 25px 80px rgba(0, 0, 0, 0.6)' }}
            >
              {/* Header - Only shown when there are messages */}
              {(messages.length > 0 || streamingResponse) && (
                <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-cyan-400/60 animate-pulse" />
                    <span className="text-xs uppercase tracking-[0.15em] text-white/40">Mind of Kevin</span>
                  </div>
                  <button
                    onClick={() => setIsChatMinimized(true)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-white/40 hover:text-white/60"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M18 15l-6-6-6 6"/>
                    </svg>
                    <span className="text-xs">Minimize</span>
                  </button>
                </div>
              )}

              {/* Welcome Title - Only when no messages */}
              {messages.length === 0 && !streamingResponse && (
                <div className="text-center py-8 px-6">
                  <p className="text-white/25 text-2xl sm:text-3xl tracking-[0.2em] sm:tracking-[0.3em] uppercase font-light">Mind of Kevin</p>
                  <p className="text-white/15 text-xs sm:text-sm mt-3 tracking-widest">Speak or type to begin</p>
                </div>
              )}

              {/* Messages Area */}
              {(messages.length > 0 || streamingResponse) && (
                <div className="max-h-[40vh] sm:max-h-[50vh] overflow-y-auto custom-scrollbar p-4 sm:p-5">
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
                                        setIsChatMinimized(true);
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
                        {/* Text messages */}
                        {msg.role !== 'images' && (
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
                    {/* Pending images - show loading state */}
                    {pendingImages.length > 0 && pendingImages.length < 3 && (
                      <div className="flex justify-center">
                        <div className="flex gap-2 p-2 sm:p-3 rounded-2xl bg-white/[0.04] border border-white/5">
                          {pendingImages.map((img, idx) => (
                            <img
                              key={idx}
                              src={img}
                              alt={`Loading ${idx + 1}`}
                              className="w-14 h-14 sm:w-16 sm:h-16 rounded-xl object-cover opacity-70"
                            />
                          ))}
                          {Array.from({ length: 3 - pendingImages.length }).map((_, idx) => (
                            <div
                              key={`loading-${idx}`}
                              className="w-14 h-14 sm:w-16 sm:h-16 rounded-xl bg-white/5 flex items-center justify-center"
                            >
                              <div className="w-5 h-5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                            </div>
                          ))}
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
              <div className={`p-4 ${messages.length > 0 || streamingResponse ? 'border-t border-white/5' : ''}`}>
                <form onSubmit={handleChatSubmit} className="relative group w-full">
                  {/* Ambient glow behind input */}
                  <div
                    className="absolute -inset-1 rounded-full bg-gradient-to-r from-cyan-500/20 via-purple-500/10 to-cyan-500/20 blur-xl opacity-50 group-focus-within:opacity-80 transition-opacity duration-500"
                    style={{ filter: 'blur(20px)' }}
                  />

                  <div className="relative flex items-center gap-3">
                    {/* Voice button */}
                    {!isConnected ? (
                      <button
                        type="button"
                        onClick={connect}
                        className="flex-shrink-0 group/mic flex items-center justify-center w-11 h-11 sm:w-12 sm:h-12 rounded-full bg-black/60 backdrop-blur-xl border border-white/15 hover:bg-black/70 hover:border-white/25 transition-all"
                        style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)' }}
                        title="Start Voice"
                      >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white/50 group-hover/mic:text-white/90 transition-colors sm:w-5 sm:h-5">
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
                        className="flex-shrink-0 group/mic relative flex items-center justify-center w-11 h-11 sm:w-12 sm:h-12 rounded-full bg-black/60 backdrop-blur-xl border border-cyan-500/30 hover:bg-black/70 transition-all"
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
                    <div className="relative flex-1">
                      <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder={isConnected ? "listening..." : "What's on your mind?"}
                        className="w-full bg-black/60 backdrop-blur-xl border border-white/15 rounded-full py-3 sm:py-4 pl-4 sm:pl-5 pr-11 sm:pr-12 text-white text-base placeholder-white/40 focus:outline-none focus:bg-black/70 focus:border-white/30 focus:shadow-lg focus:shadow-cyan-500/10 transition-all"
                        style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)' }}
                        disabled={isThinking}
                      />
                      <button
                        type="submit"
                        disabled={!inputText.trim() || isThinking}
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center bg-gradient-to-br from-cyan-500/30 to-blue-600/30 text-white/70 hover:text-white hover:from-cyan-500/50 hover:to-blue-600/50 disabled:opacity-20 disabled:hover:from-cyan-500/30 disabled:hover:to-blue-600/30 transition-all border border-white/10"
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

      {/* Navigation Arrows - Hidden on mobile, shown on larger screens */}
      <div className="hidden md:grid absolute bottom-8 right-6 z-30 pointer-events-auto grid-cols-3 gap-0.5 opacity-30 hover:opacity-60 transition-opacity">
          <div />
          <button onClick={() => moveCamera('up')} className="w-8 h-8 rounded flex items-center justify-center hover:bg-white/10 active:bg-white/20 transition-all">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M18 15l-6-6-6 6"/></svg>
          </button>
          <div />
          <button onClick={() => moveCamera('left')} className="w-8 h-8 rounded flex items-center justify-center hover:bg-white/10 active:bg-white/20 transition-all">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M15 18l-6-6 6-6"/></svg>
          </button>
          <button onClick={() => moveCamera('down')} className="w-8 h-8 rounded flex items-center justify-center hover:bg-white/10 active:bg-white/20 transition-all">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M6 9l6 6 6-6"/></svg>
          </button>
          <button onClick={() => moveCamera('right')} className="w-8 h-8 rounded flex items-center justify-center hover:bg-white/10 active:bg-white/20 transition-all">
             <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>
          </button>
      </div>

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

      {/* Floating Navigation Thought Seeds - only visible when chat is minimized or no conversation */}
      <div
        className={`absolute z-20 pointer-events-auto right-4 sm:right-8 top-1/2 -translate-y-1/2 flex flex-col gap-6 sm:gap-8 transition-all duration-500 ${
          !isChatMinimized && (messages.length > 0 || streamingResponse) ? 'opacity-0 pointer-events-none translate-x-8' : 'opacity-100 translate-x-0'
        }`}
      >
        {/* The Horizon - Speaking/Ideas */}
        <button
          onClick={() => setActivePanel(activePanel === 'horizon' ? null : 'horizon')}
          className={`group relative flex items-center justify-center sm:justify-end transition-all duration-500 ${activePanel === 'horizon' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute right-14 text-sm tracking-[0.15em] uppercase text-amber-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'horizon' ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4 group-hover:opacity-100 group-hover:translate-x-0'}`}>
            The Horizon
          </span>
          <div className={`relative w-5 h-5 sm:w-6 sm:h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 ${activePanel === 'horizon' ? 'bg-amber-400 shadow-[0_0_30px_rgba(251,191,36,0.7)] scale-125' : 'bg-amber-400/70 sm:group-hover:bg-amber-400 sm:group-hover:shadow-[0_0_25px_rgba(251,191,36,0.5)]'}`}>
            <div className={`absolute inset-0 rounded-full bg-amber-400/50 ${activePanel === 'horizon' ? 'animate-ping' : 'sm:group-hover:animate-pulse'}`} style={{ animationDuration: '2s' }} />
          </div>
        </button>

        {/* The Bridge - Connection */}
        <button
          onClick={() => setActivePanel(activePanel === 'bridge' ? null : 'bridge')}
          className={`group relative flex items-center justify-center sm:justify-end transition-all duration-500 ${activePanel === 'bridge' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute right-14 text-sm tracking-[0.15em] uppercase text-cyan-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'bridge' ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4 group-hover:opacity-100 group-hover:translate-x-0'}`}>
            The Bridge
          </span>
          <div className={`relative w-5 h-5 sm:w-6 sm:h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 ${activePanel === 'bridge' ? 'bg-cyan-400 shadow-[0_0_30px_rgba(34,211,238,0.7)] scale-125' : 'bg-cyan-400/70 sm:group-hover:bg-cyan-400 sm:group-hover:shadow-[0_0_25px_rgba(34,211,238,0.5)]'}`}>
            <div className={`absolute inset-0 rounded-full bg-cyan-400/50 ${activePanel === 'bridge' ? 'animate-ping' : 'sm:group-hover:animate-pulse'}`} style={{ animationDuration: '2s' }} />
          </div>
        </button>

        {/* The Echoes - Writings */}
        <button
          onClick={() => setActivePanel(activePanel === 'echoes' ? null : 'echoes')}
          className={`group relative flex items-center justify-center sm:justify-end transition-all duration-500 ${activePanel === 'echoes' ? 'opacity-100' : 'opacity-60 hover:opacity-100 active:opacity-100'}`}
        >
          <span className={`hidden sm:block absolute right-14 text-sm tracking-[0.15em] uppercase text-violet-200/90 whitespace-nowrap transition-all duration-300 font-medium ${activePanel === 'echoes' ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4 group-hover:opacity-100 group-hover:translate-x-0'}`}>
            The Echoes
          </span>
          <div className={`relative w-5 h-5 sm:w-6 sm:h-6 rounded-full transition-all duration-300 ease-out sm:group-hover:scale-150 ${activePanel === 'echoes' ? 'bg-violet-400 shadow-[0_0_30px_rgba(167,139,250,0.7)] scale-125' : 'bg-violet-400/70 sm:group-hover:bg-violet-400 sm:group-hover:shadow-[0_0_25px_rgba(167,139,250,0.5)]'}`}>
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
            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group">
                <p className="text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors">Keynote Speaking</p>
                <p className="text-white/40 text-xs mt-1">Conferences, corporate events, universities</p>
              </div>
              <div className="p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group">
                <p className="text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors">Workshops</p>
                <p className="text-white/40 text-xs mt-1">Interactive sessions on AI strategy & ethics</p>
              </div>
              <div className="p-3 rounded-xl bg-white/5 border border-white/5 hover:border-amber-500/30 active:border-amber-500/30 transition-colors cursor-pointer group">
                <p className="text-white/90 text-sm font-medium group-hover:text-amber-200 group-active:text-amber-200 transition-colors">Advisory</p>
                <p className="text-white/40 text-xs mt-1">Strategic guidance for organizations</p>
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

      {/* The Bridge Panel - Connection */}
      {activePanel === 'bridge' && (
        <div className="fixed inset-0 z-40 pointer-events-auto animate-fade-in sm:absolute sm:inset-auto sm:top-1/2 sm:right-20 sm:-translate-y-1/2 sm:z-20">
          {/* Mobile backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm sm:hidden" onClick={() => setActivePanel(null)} />
          <div className="absolute inset-4 sm:inset-auto sm:relative bg-black/90 sm:bg-black/80 backdrop-blur-xl rounded-2xl border border-cyan-500/20 p-5 sm:p-6 sm:w-80 shadow-2xl shadow-cyan-500/10 flex flex-col max-h-[calc(100vh-2rem)] sm:max-h-none overflow-y-auto">
            {/* Close button - mobile only */}
            <button onClick={() => setActivePanel(null)} className="absolute top-3 right-3 p-2 text-white/40 hover:text-white/70 sm:hidden">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
            </button>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.6)]" />
              <h3 className="text-cyan-200 text-sm tracking-[0.15em] uppercase">The Bridge</h3>
            </div>
            <p className="text-white/70 text-sm leading-relaxed mb-5">
              Send a thought into the void. I read every message that finds its way here.
            </p>
            <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); /* TODO: handle submit */ }}>
              <input
                type="text"
                placeholder="Your name"
                className="w-full px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40"
              />
              <input
                type="email"
                placeholder="Your email"
                className="w-full px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40"
              />
              <textarea
                placeholder="Your message..."
                rows={3}
                className="w-full px-4 py-3 sm:py-2.5 rounded-xl bg-white/5 border border-white/10 text-white text-base sm:text-sm placeholder-white/30 focus:outline-none focus:border-cyan-500/40 resize-none"
              />
              <button
                type="submit"
                className="w-full py-3 sm:py-2.5 rounded-xl bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 text-cyan-200 text-sm hover:from-cyan-500/30 hover:to-blue-500/30 active:from-cyan-500/30 active:to-blue-500/30 transition-all"
              >
                Send Into The Void
              </button>
            </form>
            <div className="mt-4 pt-4 border-t border-white/5 flex justify-center gap-6 sm:gap-4">
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-white/30 hover:text-cyan-400 active:text-cyan-400 transition-colors p-2">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="sm:w-[18px] sm:h-[18px]"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-white/30 hover:text-cyan-400 active:text-cyan-400 transition-colors p-2">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="sm:w-[18px] sm:h-[18px]"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
              </a>
            </div>
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
            <div className="space-y-3">
              <a href="#" className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group">
                <p className="text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors">When Sand Speaks</p>
                <p className="text-white/40 text-xs mt-1">A meditation on AI consciousness</p>
              </a>
              <a href="#" className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group">
                <p className="text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors">The Emma Project</p>
                <p className="text-white/40 text-xs mt-1">Preserving memory, honoring legacy</p>
              </a>
              <a href="#" className="block p-3 rounded-xl bg-white/5 border border-white/5 hover:border-violet-500/30 active:border-violet-500/30 transition-colors group">
                <p className="text-white/90 text-sm font-medium group-hover:text-violet-200 group-active:text-violet-200 transition-colors">Essays & Reflections</p>
                <p className="text-white/40 text-xs mt-1">On technology, family, and meaning</p>
              </a>
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
