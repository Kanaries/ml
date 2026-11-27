import { ImageResponse } from 'next/og';

export const alt = 'ROC-AUC and Precision-Recall Curve Calculator';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function RocAucOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', background: '#faf8f1', color: '#1c1a16', padding: 66, border: '18px solid #1d3f72', gap: 52, alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 20 }}>
        <div style={{ display: 'flex', color: '#1d3f72', fontSize: 24, fontWeight: 700 }}>FREE · ROC &amp; PR CURVES · IN-BROWSER</div>
        <div style={{ display: 'flex', fontSize: 64, lineHeight: 1.03, fontWeight: 760, letterSpacing: '-0.04em' }}>ROC-AUC &amp; PR Curve Calculator</div>
        <div style={{ display: 'flex', fontSize: 27, color: '#5d574c' }}>Threshold Slider · Youden’s J · Best F1 · Brier Score</div>
      </div>
      <div style={{ display: 'flex', width: 320, height: 320, border: '2px solid #1c1a16', background: '#fffdf8', padding: 24, alignItems: 'center', justifyContent: 'center' }}>
        <svg width="268" height="268" viewBox="0 0 268 268">
          <line x1="20" y1="248" x2="248" y2="20" stroke="#9b9387" strokeDasharray="6 6" strokeWidth="3" />
          <polygon points="20,248 20,160 54,110 96,64 162,36 248,20 248,248" fill="#1d3f72" fillOpacity="0.16" />
          <polyline points="20,248 20,160 54,110 96,64 162,36 248,20" fill="none" stroke="#1d3f72" strokeWidth="6" />
          <circle cx="96" cy="64" r="10" fill="#d9534f" stroke="#ffffff" strokeWidth="4" />
        </svg>
      </div>
    </div>,
    size,
  );
}
