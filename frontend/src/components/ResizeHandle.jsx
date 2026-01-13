import { useCallback, useEffect, useRef, useState } from 'react';
import './ResizeHandle.css';

/**
 * A draggable resize handle for panel resizing.
 *
 * @param {Object} props
 * @param {'horizontal' | 'vertical'} props.direction - Resize direction
 * @param {(delta: number) => void} props.onResize - Called with pixel delta during drag
 * @param {() => void} [props.onResizeStart] - Called when drag starts
 * @param {() => void} [props.onResizeEnd] - Called when drag ends
 * @param {number} [props.minValue] - Minimum value constraint
 * @param {number} [props.maxValue] - Maximum value constraint
 * @param {string} [props.className] - Additional CSS classes
 */
function ResizeHandle({
  direction = 'horizontal',
  onResize,
  onResizeStart,
  onResizeEnd,
  className = '',
}) {
  const [isDragging, setIsDragging] = useState(false);
  const startPosRef = useRef(0);
  const handleRef = useRef(null);

  const handleMouseDown = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(true);
      startPosRef.current = direction === 'horizontal' ? e.clientX : e.clientY;
      onResizeStart?.();
    },
    [direction, onResizeStart]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e) => {
      const currentPos = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = currentPos - startPosRef.current;
      startPosRef.current = currentPos;
      onResize(delta);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      onResizeEnd?.();
    };

    // Prevent text selection during drag
    document.body.style.userSelect = 'none';
    document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, direction, onResize, onResizeEnd]);

  return (
    <div
      ref={handleRef}
      className={`resize-handle resize-handle-${direction} ${isDragging ? 'dragging' : ''} ${className}`}
      onMouseDown={handleMouseDown}
      role="separator"
      aria-orientation={direction === 'horizontal' ? 'vertical' : 'horizontal'}
      tabIndex={0}
    >
      <div className="resize-handle-line" />
    </div>
  );
}

export default ResizeHandle;
