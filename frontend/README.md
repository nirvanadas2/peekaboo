# Peekaboo frontend

Two-page React Router site: a scroll-story landing page (`/`) and the
scan results dashboard (`/dashboard`, placeholder for now).

Self-contained Vite + React + TypeScript project. Does not touch the
Python pipeline in `../peekaboo/`.

## Stack

- React + Vite + TypeScript, react-router-dom
- Lenis (smooth scroll) + GSAP ScrollTrigger (reveal animations)
- Three.js (vanilla, via `three-stdlib` loaders) for the hero's 3D character

## Develop

```
npm install
npm run dev
```

## Attribution

The scroll-story mechanics, cursor, navbar structure, and 3D character rig
are adapted from a third-party MIT-licensed portfolio template. See
[`NOTICE.md`](./NOTICE.md) for the full license text and copyright notice.
