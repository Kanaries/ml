import { ImageResponse } from 'next/og';

export const alt = 'Logistic Regression Calculator with Odds Ratios and Decision Boundary';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function LogisticRegressionOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', background: '#faf8f1', color: '#1c1a16', padding: 66, border: '18px solid #1d3f72', gap: 50, alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 20 }}>
        <div style={{ display: 'flex', color: '#1d3f72', fontSize: 24, fontWeight: 700 }}>FREE · INTERACTIVE · IN-BROWSER</div>
        <div style={{ display: 'flex', fontSize: 64, lineHeight: 1.02, fontWeight: 760, letterSpacing: '-0.04em' }}>Logistic Regression Calculator</div>
        <div style={{ display: 'flex', fontSize: 27, color: '#5d574c' }}>Odds ratios · Sigmoid fit · Decision boundary</div>
      </div>
      <svg width="330" height="290" viewBox="0 0 330 290">
        <rect width="330" height="290" fill="#fffdf8" stroke="#1c1a16" strokeWidth="2" />
        <path d="M 30 244 C 100 244, 105 210, 150 150 S 225 45, 300 45" fill="none" stroke="#1d3f72" strokeWidth="8" />
        <line x1="25" y1="145" x2="305" y2="145" stroke="#9b9487" strokeWidth="2" strokeDasharray="8 8" />
        {[42, 70, 99, 135, 176, 218, 260, 286].map((x, index) => <circle key={x} cx={x} cy={index < 4 ? 245 : 45} r="8" fill={index < 4 ? '#2b7a78' : '#d45d4c'} />)}
      </svg>
    </div>,
    size,
  );
}
