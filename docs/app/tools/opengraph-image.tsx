import { ImageResponse } from 'next/og';

export const alt = '@kanaries/ml browser-native machine learning tools';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function ToolsOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', background: '#faf8f1', color: '#1c1a16', padding: '64px 72px', border: '18px solid #1d3f72' }}>
      <div style={{ display: 'flex', fontSize: 28, fontWeight: 700, letterSpacing: '-0.02em' }}>@kanaries/ml · Browser-native ML</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
        <div style={{ display: 'flex', maxWidth: 980, fontSize: 72, lineHeight: 1.04, fontWeight: 750, letterSpacing: '-0.04em' }}>Free machine learning calculators & visualizations</div>
        <div style={{ display: 'flex', fontSize: 29, color: '#5d574c' }}>Private client-side calculation · JavaScript and Python code · No sign-in</div>
      </div>
    </div>,
    size,
  );
}
