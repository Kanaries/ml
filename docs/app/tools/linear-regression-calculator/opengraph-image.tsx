import { ImageResponse } from 'next/og';

export const alt = 'Linear and Ridge Regression CSV Calculator';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function LinearRegressionOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', background: '#faf8f1', color: '#1c1a16', padding: 66, border: '18px solid #1d3f72', gap: 52, alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 20 }}>
        <div style={{ display: 'flex', color: '#1d3f72', fontSize: 24, fontWeight: 700 }}>FREE · OLS &amp; RIDGE · CSV IN-BROWSER</div>
        <div style={{ display: 'flex', fontSize: 62, lineHeight: 1.03, fontWeight: 760, letterSpacing: '-0.04em' }}>Linear &amp; Ridge Regression Calculator</div>
        <div style={{ display: 'flex', fontSize: 27, color: '#5d574c' }}>R² · Adjusted R² · RMSE · Coefficients · Residual Plot</div>
      </div>
      <div style={{ display: 'flex', width: 320, height: 320, border: '2px solid #1c1a16', background: '#fffdf8', padding: 24, alignItems: 'center', justifyContent: 'center' }}>
        <svg width="268" height="268" viewBox="0 0 268 268">
          <line x1="20" y1="244" x2="248" y2="244" stroke="#9b9387" strokeWidth="3" />
          <line x1="20" y1="20" x2="20" y2="244" stroke="#9b9387" strokeWidth="3" />
          <line x1="24" y1="224" x2="244" y2="38" stroke="#1d3f72" strokeWidth="6" />
          <circle cx="48" cy="212" r="8" fill="#1d3f72" />
          <circle cx="84" cy="168" r="8" fill="#1d3f72" />
          <circle cx="122" cy="150" r="8" fill="#1d3f72" />
          <circle cx="158" cy="104" r="8" fill="#1d3f72" />
          <circle cx="196" cy="84" r="8" fill="#1d3f72" />
          <circle cx="230" cy="48" r="8" fill="#1d3f72" />
        </svg>
      </div>
    </div>,
    size,
  );
}
