import { useCallback, useRef, useState } from "react";

/**
 * Voice input: browser first, backend second.
 *
 * Where the Web Speech API exists (Chrome/Edge) it transcribes `he-IL` locally
 * and the audio never leaves the machine. Where it doesn't (Firefox, older
 * Safari) we record with MediaRecorder and POST the blob to the bridge, which
 * needs a backend STT model installed — and says so plainly when there isn't
 * one. There is no third path that quietly returns nothing.
 */

interface SpeechAlternative {
  transcript: string;
}
interface SpeechResult {
  isFinal: boolean;
  0: SpeechAlternative;
}
interface SpeechEvent {
  resultIndex: number;
  results: { length: number; [index: number]: SpeechResult };
}
interface SpeechRecognizer {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start(): void;
  stop(): void;
  onresult: ((event: SpeechEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
}
type RecognizerCtor = new () => SpeechRecognizer;

function recognizerCtor(): RecognizerCtor | undefined {
  const scope = window as unknown as {
    SpeechRecognition?: RecognizerCtor;
    webkitSpeechRecognition?: RecognizerCtor;
  };
  return scope.SpeechRecognition ?? scope.webkitSpeechRecognition;
}

export type VoiceMode = "speech-api" | "recorder" | "unsupported";

export interface VoiceHandlers {
  /** Live partial text, for showing in the textarea as the user speaks. */
  onInterim?: (text: string) => void;
  /** A finished browser-side transcript. */
  onTranscript?: (text: string) => void;
  /** A recorded clip, to be transcribed by the bridge. */
  onRecording?: (audio: Blob) => void;
}

export function voiceMode(): VoiceMode {
  if (recognizerCtor()) return "speech-api";
  if (typeof MediaRecorder !== "undefined" && navigator.mediaDevices) return "recorder";
  return "unsupported";
}

export function useVoiceInput(handlers: VoiceHandlers) {
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const recognizerRef = useRef<SpeechRecognizer | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const mode = voiceMode();

  const stop = useCallback(() => {
    recognizerRef.current?.stop();
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }, []);

  const start = useCallback(async () => {
    setError(null);

    const Recognizer = recognizerCtor();
    if (Recognizer) {
      const recognizer = new Recognizer();
      recognizer.lang = "he-IL";
      recognizer.interimResults = true;
      recognizer.continuous = false;
      recognizer.onresult = (event) => {
        let interim = "";
        let final = "";
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const result = event.results[i];
          if (result.isFinal) final += result[0].transcript;
          else interim += result[0].transcript;
        }
        if (interim) handlers.onInterim?.(interim.trim());
        if (final) handlers.onTranscript?.(final.trim());
      };
      recognizer.onerror = (event) =>
        setError(
          event.error === "not-allowed"
            ? "הגישה למיקרופון נדחתה על-ידי הדפדפן"
            : `זיהוי הדיבור נכשל: ${event.error}`,
        );
      recognizer.onend = () => {
        setActive(false);
        recognizerRef.current = null;
      };
      recognizerRef.current = recognizer;
      recognizer.start();
      setActive(true);
      return;
    }

    if (mode === "unsupported") {
      setError("הדפדפן הזה לא תומך בקלט קולי");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (exc) {
      setError(
        `אין גישה למיקרופון: ${exc instanceof Error ? exc.message : String(exc)}`,
      );
      return;
    }

    // An AnalyserNode drives the waveform, so the user can see that the
    // microphone is actually picking something up.
    const context = new AudioContext();
    const node = context.createAnalyser();
    node.fftSize = 512;
    context.createMediaStreamSource(stream).connect(node);
    setAnalyser(node);

    const recorder = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.onstop = () => {
      cleanupRef.current?.();
      if (chunks.length > 0) {
        handlers.onRecording?.(new Blob(chunks, { type: recorder.mimeType }));
      }
    };
    cleanupRef.current = () => {
      stream.getTracks().forEach((track) => track.stop());
      void context.close();
      setAnalyser(null);
      setActive(false);
      recorderRef.current = null;
    };

    recorderRef.current = recorder;
    recorder.start();
    setActive(true);
  }, [handlers, mode]);

  return { mode, active, error, analyser, start, stop, clearError: () => setError(null) };
}
