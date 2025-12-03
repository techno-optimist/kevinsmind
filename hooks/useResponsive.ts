import { useState, useEffect, useCallback, useMemo } from 'react';

/**
 * Breakpoint definitions matching Tailwind defaults
 */
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

export type Breakpoint = keyof typeof BREAKPOINTS;

/**
 * Screen size category for simplified responsive logic
 */
export type ScreenSize = 'mobile' | 'tablet' | 'desktop' | 'wide';

/**
 * Device orientation
 */
export type Orientation = 'portrait' | 'landscape';

/**
 * Responsive state returned by the hook
 */
export interface ResponsiveState {
  // Current dimensions
  width: number;
  height: number;

  // Breakpoint checks (true if viewport >= breakpoint)
  isSm: boolean;
  isMd: boolean;
  isLg: boolean;
  isXl: boolean;
  is2xl: boolean;

  // Convenience aliases
  isMobile: boolean;      // < sm (640px)
  isTablet: boolean;      // >= sm && < lg
  isDesktop: boolean;     // >= lg
  isWide: boolean;        // >= xl

  // Simplified screen size
  screenSize: ScreenSize;

  // Orientation
  orientation: Orientation;
  isPortrait: boolean;
  isLandscape: boolean;

  // Device pixel ratio (for high-DPI displays)
  pixelRatio: number;
  isRetina: boolean;

  // Touch capability
  hasTouch: boolean;

  // Safe area insets (for notched devices)
  safeAreaTop: number;
  safeAreaBottom: number;
  safeAreaLeft: number;
  safeAreaRight: number;
}

/**
 * Parse CSS env() value to number
 */
const parseSafeAreaInset = (property: string): number => {
  if (typeof window === 'undefined' || typeof getComputedStyle === 'undefined') {
    return 0;
  }

  const testEl = document.createElement('div');
  testEl.style.paddingTop = `env(${property}, 0px)`;
  document.body.appendChild(testEl);
  const value = parseInt(getComputedStyle(testEl).paddingTop, 10) || 0;
  document.body.removeChild(testEl);
  return value;
};

/**
 * Get current responsive state
 */
const getResponsiveState = (): ResponsiveState => {
  const width = typeof window !== 'undefined' ? window.innerWidth : 0;
  const height = typeof window !== 'undefined' ? window.innerHeight : 0;
  const pixelRatio = typeof window !== 'undefined' ? window.devicePixelRatio : 1;

  // Breakpoint checks
  const isSm = width >= BREAKPOINTS.sm;
  const isMd = width >= BREAKPOINTS.md;
  const isLg = width >= BREAKPOINTS.lg;
  const isXl = width >= BREAKPOINTS.xl;
  const is2xl = width >= BREAKPOINTS['2xl'];

  // Convenience aliases
  const isMobile = !isSm;
  const isTablet = isSm && !isLg;
  const isDesktop = isLg;
  const isWide = isXl;

  // Screen size category
  let screenSize: ScreenSize = 'mobile';
  if (is2xl) screenSize = 'wide';
  else if (isLg) screenSize = 'desktop';
  else if (isSm) screenSize = 'tablet';

  // Orientation
  const isPortrait = height > width;
  const isLandscape = width >= height;
  const orientation: Orientation = isPortrait ? 'portrait' : 'landscape';

  // Touch capability
  const hasTouch = typeof window !== 'undefined' && (
    'ontouchstart' in window ||
    navigator.maxTouchPoints > 0
  );

  return {
    width,
    height,
    isSm,
    isMd,
    isLg,
    isXl,
    is2xl,
    isMobile,
    isTablet,
    isDesktop,
    isWide,
    screenSize,
    orientation,
    isPortrait,
    isLandscape,
    pixelRatio,
    isRetina: pixelRatio >= 2,
    hasTouch,
    safeAreaTop: 0, // Updated on mount
    safeAreaBottom: 0,
    safeAreaLeft: 0,
    safeAreaRight: 0,
  };
};

/**
 * React hook for responsive design
 *
 * @example
 * ```tsx
 * const { isMobile, isDesktop, screenSize } = useResponsive();
 *
 * return (
 *   <div className={isMobile ? 'p-2' : 'p-6'}>
 *     {isDesktop && <Sidebar />}
 *     <MainContent />
 *   </div>
 * );
 * ```
 */
export function useResponsive(): ResponsiveState {
  const [state, setState] = useState<ResponsiveState>(getResponsiveState);

  useEffect(() => {
    // Update safe area insets on mount
    const updateSafeAreas = () => {
      setState(prev => ({
        ...prev,
        safeAreaTop: parseSafeAreaInset('safe-area-inset-top'),
        safeAreaBottom: parseSafeAreaInset('safe-area-inset-bottom'),
        safeAreaLeft: parseSafeAreaInset('safe-area-inset-left'),
        safeAreaRight: parseSafeAreaInset('safe-area-inset-right'),
      }));
    };

    updateSafeAreas();

    // Handle resize with debounce
    let timeoutId: ReturnType<typeof setTimeout>;
    const handleResize = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        setState({ ...getResponsiveState() });
        updateSafeAreas();
      }, 100);
    };

    // Handle orientation change
    const handleOrientationChange = () => {
      // Small delay to let the browser settle
      setTimeout(() => {
        setState({ ...getResponsiveState() });
        updateSafeAreas();
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleOrientationChange);

    // Also listen for visual viewport changes (mobile browser chrome)
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleResize);
    }

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
      if (window.visualViewport) {
        window.visualViewport.removeEventListener('resize', handleResize);
      }
    };
  }, []);

  return state;
}

/**
 * Hook for matching a specific breakpoint
 *
 * @example
 * ```tsx
 * const isLargeScreen = useBreakpoint('lg');
 * ```
 */
export function useBreakpoint(breakpoint: Breakpoint): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.innerWidth >= BREAKPOINTS[breakpoint];
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia(`(min-width: ${BREAKPOINTS[breakpoint]}px)`);

    const handleChange = (e: MediaQueryListEvent) => {
      setMatches(e.matches);
    };

    // Initial check
    setMatches(mediaQuery.matches);

    // Modern API
    mediaQuery.addEventListener('change', handleChange);

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, [breakpoint]);

  return matches;
}

/**
 * Hook for media query matching
 *
 * @example
 * ```tsx
 * const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');
 * const prefersDark = useMediaQuery('(prefers-color-scheme: dark)');
 * ```
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);

    const handleChange = (e: MediaQueryListEvent) => {
      setMatches(e.matches);
    };

    setMatches(mediaQuery.matches);
    mediaQuery.addEventListener('change', handleChange);

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, [query]);

  return matches;
}

/**
 * Returns a value based on current screen size
 *
 * @example
 * ```tsx
 * const columns = useResponsiveValue({
 *   mobile: 1,
 *   tablet: 2,
 *   desktop: 3,
 *   wide: 4,
 * });
 * ```
 */
export function useResponsiveValue<T>(values: Partial<Record<ScreenSize, T>>): T | undefined {
  const { screenSize } = useResponsive();

  return useMemo(() => {
    // Try exact match first
    if (values[screenSize] !== undefined) {
      return values[screenSize];
    }

    // Fall back to smaller sizes
    const sizes: ScreenSize[] = ['mobile', 'tablet', 'desktop', 'wide'];
    const currentIndex = sizes.indexOf(screenSize);

    for (let i = currentIndex - 1; i >= 0; i--) {
      if (values[sizes[i]] !== undefined) {
        return values[sizes[i]];
      }
    }

    return undefined;
  }, [screenSize, values]);
}

/**
 * Returns dynamic viewport height accounting for mobile browser chrome
 * This is useful when 100dvh isn't available or you need the value in JS
 */
export function useDynamicViewportHeight(): number {
  const [height, setHeight] = useState(() => {
    if (typeof window === 'undefined') return 0;
    return window.visualViewport?.height ?? window.innerHeight;
  });

  useEffect(() => {
    const updateHeight = () => {
      setHeight(window.visualViewport?.height ?? window.innerHeight);
    };

    updateHeight();

    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', updateHeight);
    }
    window.addEventListener('resize', updateHeight);

    return () => {
      if (window.visualViewport) {
        window.visualViewport.removeEventListener('resize', updateHeight);
      }
      window.removeEventListener('resize', updateHeight);
    };
  }, []);

  return height;
}

export default useResponsive;
