import { ImageResponse } from 'next/og';

export const alt = '@kanaries/ml interactive machine learning playgrounds';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function PlaygroundOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', background: '#faf8f1', color: '#1c1a16', padding: '64px 72px', border: '18px solid #1d3f72' }}>
      <div style={{ display: 'flex', fontSize: 28, fontWeight: 700 }}>@kanaries/ml · Interactive learning</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
        <div style={{ display: 'flex', fontSize: 76, lineHeight: 1.02, fontWeight: 760, letterSpacing: '-0.04em' }}>Machine learning playgrounds</div>
        <div style={{ display: 'flex', fontSize: 30, color: '#5d574c' }}>PCA · KNN · Gradient Descent · K-Means</div>
      </div>
    </div>,
    size,
  );
}
