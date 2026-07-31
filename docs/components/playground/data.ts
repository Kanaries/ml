export type Point2D = { x: number; y: number; label?: number };

export const COLORS = ['#2b7a78', '#d45d4c', '#7b61a8', '#d39b2a', '#3480b8', '#8a6d3b'];

function seededRandom(seed = 17) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(random: () => number) {
  const u = Math.max(random(), 1e-12);
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makeBlobs(count = 72, centers = [[-1.8, -1.2], [1.7, -0.5], [0.2, 1.8]], spread = 0.48, seed = 11): Point2D[] {
  const random = seededRandom(seed);
  return Array.from({ length: count }, (_, index) => {
    const label = index % centers.length;
    return { x: centers[label][0] + gaussian(random) * spread, y: centers[label][1] + gaussian(random) * spread, label };
  });
}

export function makeMoons(count = 80, noise = 0.1, seed = 23): Point2D[] {
  const random = seededRandom(seed);
  return Array.from({ length: count }, (_, index) => {
    const label = index % 2;
    const t = (Math.floor(index / 2) / Math.max(1, Math.ceil(count / 2) - 1)) * Math.PI;
    if (label === 0) return { x: Math.cos(t) + gaussian(random) * noise, y: Math.sin(t) + gaussian(random) * noise, label };
    return { x: 1 - Math.cos(t) + gaussian(random) * noise, y: 0.45 - Math.sin(t) + gaussian(random) * noise, label };
  });
}

export function makeXor(count = 72, seed = 41): Point2D[] {
  const random = seededRandom(seed);
  return Array.from({ length: count }, () => {
    const x = random() * 5 - 2.5;
    const y = random() * 5 - 2.5;
    return { x, y, label: x * y >= 0 ? 0 : 1 };
  });
}

export function extent(values: number[], padding = 0.12): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = Math.max((max - min) * padding, 0.2);
  return [min - pad, max + pad];
}
