"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

import {
  createFrameFormData,
  parsePreprocessResponse,
  preprocessUrl,
  type PreprocessResponse,
} from "../lib/live-detection";

export type CameraStatus = "Idle" | "Starting" | "Active" | "Stopping" | "Error";
export type ProcessingStatus = "Idle" | "Processing" | "Error";
export type BackendStatus = "Not checked" | "Connected" | "Unavailable";

const FRAME_INTERVAL_MS = 1000;
const REQUEST_TIMEOUT_MS = 15_000;

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "Camera permission was denied. Allow camera access in your browser and try again.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "No camera was found. Connect a camera and try again.";
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return "The camera is unavailable or already in use by another application.";
    }
  }
  return "Could not start the camera. Check browser permissions and camera availability.";
}

function apiErrorMessage(error: unknown, timedOut: boolean): string {
  if (timedOut) {
    return "The preprocessing request timed out. Check that the local backend is running.";
  }
  if (error instanceof TypeError) {
    return "Could not reach the local preprocessing backend. Check that it is running and CORS is configured.";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "The preprocessing request failed.";
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    track.onended = null;
    track.stop();
  }
}

export function useLivePreprocessing(
  videoRef: RefObject<HTMLVideoElement | null>,
  captureCanvasRef: RefObject<HTMLCanvasElement | null>,
) {
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("Idle");
  const [processingStatus, setProcessingStatus] =
    useState<ProcessingStatus>("Idle");
  const [backendStatus, setBackendStatus] =
    useState<BackendStatus>("Not checked");
  const [response, setResponse] = useState<PreprocessResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [permissionRequestPending, setPermissionRequestPending] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef(0);
  const pendingStartRef = useRef(false);
  const inFlightRef = useRef(false);
  const frameIndexRef = useRef(0);
  const frameTimerRef = useRef<number | null>(null);
  const idleTimerRef = useRef<number | null>(null);
  const requestControllerRef = useRef<AbortController | null>(null);
  const requestTimeoutRef = useRef<number | null>(null);
  const mountedRef = useRef(false);

  const clearActiveResources = useCallback(() => {
    if (frameTimerRef.current !== null) {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    if (requestTimeoutRef.current !== null) {
      window.clearTimeout(requestTimeoutRef.current);
      requestTimeoutRef.current = null;
    }
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    inFlightRef.current = false;

    const stream = streamRef.current;
    streamRef.current = null;
    if (stream) {
      stopTracks(stream);
    }

    const video = videoRef.current;
    if (video && (!stream || video.srcObject === stream)) {
      video.pause();
      video.srcObject = null;
    }

    const canvas = captureCanvasRef.current;
    const context = canvas?.getContext("2d");
    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [captureCanvasRef, videoRef]);

  const processFrame = useCallback(
    async (sessionId: number) => {
      if (inFlightRef.current || sessionIdRef.current !== sessionId) {
        return;
      }

      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.videoWidth <= 0 || video.videoHeight <= 0) {
        return;
      }

      const context = canvas.getContext("2d");
      if (!context) {
        if (mountedRef.current && sessionIdRef.current === sessionId) {
          setProcessingStatus("Error");
          setErrorMessage("Could not capture a frame from the camera.");
        }
        return;
      }

      inFlightRef.current = true;
      setProcessingStatus("Processing");
      setErrorMessage(null);
      let requestFailed = false;

      try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) => {
          canvas.toBlob(resolve, "image/jpeg", 0.85);
        });
        if (sessionIdRef.current !== sessionId) {
          return;
        }
        if (!blob) {
          throw new Error("Could not encode the camera frame.");
        }

        const controller = new AbortController();
        let timedOut = false;
        requestControllerRef.current = controller;
        requestTimeoutRef.current = window.setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, REQUEST_TIMEOUT_MS);

        try {
          const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
          const apiResponse = await fetch(preprocessUrl(baseUrl), {
            method: "POST",
            body: createFrameFormData(blob, frameIndexRef.current++),
            signal: controller.signal,
            cache: "no-store",
          });
          const payload: unknown = await apiResponse.json();
          if (!apiResponse.ok) {
            const detail =
              payload &&
              typeof payload === "object" &&
              "detail" in payload &&
              typeof payload.detail === "string"
                ? payload.detail
                : "The preprocessing backend rejected the frame.";
            throw new Error(detail);
          }
          const parsed = parsePreprocessResponse(payload);
          if (sessionIdRef.current !== sessionId) {
            return;
          }
          setResponse(parsed);
          setBackendStatus("Connected");
          setProcessingStatus("Idle");
          setErrorMessage(null);
        } catch (error) {
          if (sessionIdRef.current !== sessionId) {
            return;
          }
          if (controller.signal.aborted && !timedOut) {
            return;
          }
          setBackendStatus("Unavailable");
          setProcessingStatus("Error");
          setErrorMessage(apiErrorMessage(error, timedOut));
          requestFailed = true;
        } finally {
          if (requestControllerRef.current === controller) {
            requestControllerRef.current = null;
            if (requestTimeoutRef.current !== null) {
              window.clearTimeout(requestTimeoutRef.current);
              requestTimeoutRef.current = null;
            }
          }
        }
      } catch (error) {
        if (sessionIdRef.current === sessionId) {
          requestFailed = true;
          setProcessingStatus("Error");
          setErrorMessage(
            error instanceof Error
              ? error.message
              : "Could not process the camera frame.",
          );
        }
      } finally {
        if (sessionIdRef.current === sessionId) {
          inFlightRef.current = false;
          if (!requestFailed) {
            setProcessingStatus((current) =>
              current === "Processing" ? "Idle" : current,
            );
          }
        }
      }
    },
    [captureCanvasRef, videoRef],
  );

  const stopCamera = useCallback(
    (nextStatus: "Idle" | "Error" = "Idle", reason?: string) => {
      sessionIdRef.current += 1;
      if (idleTimerRef.current !== null) {
        window.clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
      }
      setCameraStatus("Stopping");
      clearActiveResources();
      setResponse(null);
      setProcessingStatus("Idle");
      setBackendStatus("Not checked");
      setErrorMessage(reason ?? null);
      if (nextStatus === "Error") {
        setCameraStatus("Error");
      } else {
        idleTimerRef.current = window.setTimeout(() => {
          idleTimerRef.current = null;
          if (mountedRef.current) {
            setCameraStatus("Idle");
          }
        }, 120);
      }
    },
    [clearActiveResources],
  );

  const startCamera = useCallback(async () => {
    if (streamRef.current || pendingStartRef.current) {
      return;
    }

    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
    const sessionId = ++sessionIdRef.current;
    pendingStartRef.current = true;
    setPermissionRequestPending(true);
    setCameraStatus("Starting");
    setErrorMessage(null);
    setResponse(null);
    setBackendStatus("Not checked");
    setProcessingStatus("Idle");
    frameIndexRef.current = 0;

    try {
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices?.getUserMedia
      ) {
        throw new Error("Camera access is not available in this browser context.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user" },
      });
      if (sessionIdRef.current !== sessionId || !mountedRef.current) {
        stopTracks(stream);
        return;
      }

      const video = videoRef.current;
      if (!video) {
        stopTracks(stream);
        throw new Error("The camera preview is unavailable.");
      }

      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      if (sessionIdRef.current !== sessionId) {
        return;
      }

      for (const track of stream.getVideoTracks()) {
        track.onended = () => {
          if (sessionIdRef.current === sessionId) {
            stopCamera("Error", "The camera disconnected during the session.");
          }
        };
      }

      setCameraStatus("Active");
      void processFrame(sessionId);
      frameTimerRef.current = window.setInterval(() => {
        void processFrame(sessionId);
      }, FRAME_INTERVAL_MS);
    } catch (error) {
      if (sessionIdRef.current === sessionId && mountedRef.current) {
        clearActiveResources();
        setCameraStatus("Error");
        setErrorMessage(
          error instanceof Error &&
          error.message === "Camera access is not available in this browser context."
            ? error.message
            : cameraErrorMessage(error),
        );
      }
    } finally {
      pendingStartRef.current = false;
      if (mountedRef.current) {
        setPermissionRequestPending(false);
      }
    }
  }, [clearActiveResources, processFrame, stopCamera, videoRef]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      sessionIdRef.current += 1;
      if (idleTimerRef.current !== null) {
        window.clearTimeout(idleTimerRef.current);
      }
      clearActiveResources();
    };
  }, [clearActiveResources]);

  return {
    cameraStatus,
    processingStatus,
    backendStatus,
    response,
    errorMessage,
    permissionRequestPending,
    startCamera,
    stopCamera,
  };
}
