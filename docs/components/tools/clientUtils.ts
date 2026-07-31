export function downloadText(filename: string, content: string, type = 'text/csv;charset=utf-8') {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function downloadSvgAsPng(svg: SVGSVGElement | null, filename: string) {
  if (!svg) return;
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const sourceElements = [svg, ...Array.from(svg.querySelectorAll('*'))];
  const clonedElements = [clone, ...Array.from(clone.querySelectorAll('*'))];
  sourceElements.forEach((sourceElement, index) => {
    const targetElement = clonedElements[index] as SVGElement | undefined;
    if (!targetElement) return;
    const computed = getComputedStyle(sourceElement);
    const properties = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray', 'font-family', 'font-size', 'font-weight'];
    const inlined = properties
      .map((property) => `${property}:${computed.getPropertyValue(property)}`)
      .join(';');
    targetElement.setAttribute('style', inlined);
  });
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const viewBox = svg.viewBox.baseVal;
  const width = Math.max(600, viewBox.width * 2);
  const height = Math.max(400, viewBox.height * 2);
  const source = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.fillStyle = '#fffdf8';
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    URL.revokeObjectURL(url);
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };
  image.src = url;
}
