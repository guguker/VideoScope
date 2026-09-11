/**
 * Convert a client-space click into intrinsic video coordinates. The video is
 * rendered with object-fit: contain, so clicks in the surrounding letterbox
 * are deliberately rejected.
 */
export function pointFromClient({ clientX, clientY, rect, videoWidth, videoHeight }) {
  if (!rect || rect.width <= 0 || rect.height <= 0 || videoWidth <= 0 || videoHeight <= 0) return null;
  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  const left = rect.left + (rect.width - width) / 2;
  const top = rect.top + (rect.height - height) / 2;
  if (clientX < left || clientX > left + width || clientY < top || clientY > top + height) return null;
  const clean = value => Number(Math.max(0, Math.min(1, value)).toFixed(6));
  return { x: clean((clientX - left) / width), y: clean((clientY - top) / height) };
}

export function pointToStyle(point, rect, videoWidth, videoHeight) {
  if (!point || point.x == null || point.y == null || !rect || videoWidth <= 0 || videoHeight <= 0) return null;
  const scale = Math.min(rect.width / videoWidth, rect.height / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  const left = (rect.width - width) / 2 + point.x * width;
  const top = (rect.height - height) / 2 + point.y * height;
  return { left: `${left}px`, top: `${top}px` };
}
