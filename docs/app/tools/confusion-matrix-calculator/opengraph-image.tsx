import { ImageResponse } from 'next/og';

export const alt = 'Confusion Matrix and F1 Calculator';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export default function ConfusionMatrixOpenGraphImage() {
  return new ImageResponse(
    <div style={{ width: '100%', height: '100%', display: 'flex', background: '#faf8f1', color: '#1c1a16', padding: 66, border: '18px solid #1d3f72', gap: 58, alignItems: 'center' }}>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, gap: 20 }}>
        <div style={{ display: 'flex', color: '#1d3f72', fontSize: 24, fontWeight: 700 }}>FREE · MULTICLASS · IN-BROWSER</div>
        <div style={{ display: 'flex', fontSize: 66, lineHeight: 1.02, fontWeight: 760, letterSpacing: '-0.04em' }}>Confusion Matrix & F1 Calculator</div>
        <div style={{ display: 'flex', fontSize: 27, color: '#5d574c' }}>Precision · Recall · MCC · Cohen’s kappa</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 18, border: '2px solid #1c1a16', background: '#fffdf8' }}>
        {[[42, 6], [9, 63]].map((row, rowIndex) => (
          <div key={rowIndex} style={{ display: 'flex', gap: 8 }}>
            {row.map((value, columnIndex) => <div key={columnIndex} style={{ width: 112, height: 112, display: 'flex', alignItems: 'center', justifyContent: 'center', background: rowIndex === columnIndex ? '#1d3f72' : '#dfe6ee', color: rowIndex === columnIndex ? '#fff' : '#1c1a16', fontSize: 42, fontWeight: 700 }}>{value}</div>)}
          </div>
        ))}
      </div>
    </div>,
    size,
  );
}
