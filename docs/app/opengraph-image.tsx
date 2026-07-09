import { ImageResponse } from 'next/og';

export const alt = '@kanaries/ml — Machine Learning in JavaScript & TypeScript';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '72px 80px',
          background: 'linear-gradient(135deg, #0f1117 0%, #1a1e2e 100%)',
          color: '#f4f4f5',
          fontFamily: 'monospace',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 30, color: '#8b93a7' }}>
          <div
            style={{
              display: 'flex',
              padding: '6px 18px',
              border: '2px solid #3d4560',
              borderRadius: 8,
              color: '#ffb86c',
            }}
          >
            npm install @kanaries/ml
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'flex', fontSize: 72, fontWeight: 700, lineHeight: 1.1 }}>
            Machine Learning in JavaScript
          </div>
          <div style={{ display: 'flex', fontSize: 36, color: '#a5adc4' }}>
            the scikit-learn way — browser & Node.js
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 28,
            color: '#8b93a7',
          }}
        >
          <div style={{ display: 'flex', fontWeight: 700, color: '#f4f4f5' }}>@kanaries/ml</div>
          <div style={{ display: 'flex' }}>ml.kanaries.net</div>
        </div>
      </div>
    ),
    { ...size },
  );
}
