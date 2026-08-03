import { useEffect, useRef } from "react";

/** A live level meter while recording. Without it a MediaRecorder session is
 *  indistinguishable from a broken microphone until the transcript comes back
 *  (or doesn't). */
export function Waveform({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!analyser || !canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const samples = new Uint8Array(analyser.frequencyBinCount);
    let frame = 0;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(samples);

      const { width, height } = canvas;
      context.clearRect(0, 0, width, height);
      context.lineWidth = 1.5;
      context.strokeStyle = "#4f46e5";
      context.beginPath();
      const step = width / samples.length;
      for (let i = 0; i < samples.length; i += 1) {
        const y = (samples[i] / 128) * (height / 2);
        const x = i * step;
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    };
    draw();

    return () => cancelAnimationFrame(frame);
  }, [analyser]);

  if (!analyser) return null;
  return (
    <canvas
      ref={canvasRef}
      width={220}
      height={28}
      className="rounded bg-[var(--color-accent-soft)]"
    />
  );
}
