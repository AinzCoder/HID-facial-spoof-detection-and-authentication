"use client";

import { useEffect, useRef } from "react";
import {
  Activity,
  Camera,
  CircleAlert,
  Gauge,
  LoaderCircle,
  ScanFace,
  ShieldCheck,
  Wifi,
  WifiOff,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { PageHeading, Panel, StatusPill } from "@/components/ui";
import { drawFaceOverlay } from "@/lib/face-overlay";
import { formatQualityFlag } from "@/lib/live-detection";
import { useLivePreprocessing } from "@/hooks/use-live-preprocessing";

export default function LiveDetectionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const {
    cameraStatus,
    processingStatus,
    backendStatus,
    response,
    errorMessage,
    permissionRequestPending,
    startCamera,
    stopCamera,
  } = useLivePreprocessing(videoRef, captureCanvasRef);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) {
      return;
    }

    const draw = () => {
      drawFaceOverlay(
        canvas,
        response?.image_dimensions ?? null,
        response?.faces ?? [],
        true,
      );
    };

    draw();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", draw);
      return () => window.removeEventListener("resize", draw);
    }
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [response]);

  const cameraTone =
    cameraStatus === "Active"
      ? "teal"
      : cameraStatus === "Error"
        ? "amber"
        : "neutral";
  const backendTone =
    backendStatus === "Connected"
      ? "teal"
      : backendStatus === "Unavailable"
        ? "amber"
        : "neutral";
  const cameraIsActive = cameraStatus === "Active";
  const warnings = response?.quality_flags ?? [];

  return (
    <AppShell>
      <div className="page-stack">
        <PageHeading
          eyebrow="Capture workspace"
          title="Live Detection"
          description="Start a local camera session to preview frames and send one frame per second to the configured preprocessing backend. No anti-spoof classification is performed."
          action={<StatusPill tone={cameraTone}>{cameraStatus}</StatusPill>}
        />

        <div className="split-grid">
          <Panel
            title="Camera preview"
            description="Camera access is requested only after you start a session. Preview frames are mirrored; submitted frames are not."
            icon={Camera}
            trailing={<StatusPill tone={backendTone}>{backendStatus}</StatusPill>}
          >
            <div className="camera-stage live-camera-stage">
              <span className="stage-status live-stage-status">
                <StatusPill tone={cameraTone}>
                  {cameraStatus === "Starting"
                    ? "Requesting camera"
                    : cameraStatus === "Active"
                      ? "Camera active"
                      : cameraStatus === "Stopping"
                        ? "Stopping camera"
                        : cameraStatus === "Error"
                          ? "Camera error"
                          : "No active stream"}
                </StatusPill>
              </span>
              <video
                ref={videoRef}
                className="camera-video"
                autoPlay
                playsInline
                muted
                aria-label="Live webcam preview"
                style={{ display: cameraIsActive ? "block" : "none" }}
              />
              <canvas
                ref={overlayCanvasRef}
                className="camera-overlay"
                aria-hidden="true"
              />
              {!cameraIsActive ? (
                <div className="camera-placeholder camera-placeholder-live">
                  <span className="camera-placeholder-icon">
                    {cameraStatus === "Starting" ? (
                      <LoaderCircle
                        size={27}
                        className="loading-icon"
                        aria-hidden="true"
                      />
                    ) : (
                      <Camera size={27} strokeWidth={1.6} aria-hidden="true" />
                    )}
                  </span>
                  <h2>
                    {cameraStatus === "Starting"
                      ? "Waiting for camera permission"
                      : cameraStatus === "Error"
                        ? "Camera unavailable"
                        : "Camera preview will appear here"}
                  </h2>
                  <p>
                    {cameraStatus === "Error"
                      ? "Review the message below, then check your browser permission and camera connection."
                      : "Your browser camera is not active. The session begins only when you select Start Camera."}
                  </p>
                </div>
              ) : null}
              <span className="stage-corner-label">
                {cameraIsActive ? "LOCAL CAMERA · LIVE" : "LOCAL SESSION · INACTIVE"}
              </span>
            </div>
            <canvas
              ref={captureCanvasRef}
              className="capture-canvas"
              aria-hidden="true"
            />
            <div className="live-controls">
              <button
                className="button button-primary"
                type="button"
                onClick={() => void startCamera()}
                disabled={
                  cameraIsActive || cameraStatus === "Starting" || permissionRequestPending
                }
              >
                {cameraStatus === "Starting" ? (
                  <LoaderCircle
                    size={16}
                    className="loading-icon"
                    aria-hidden="true"
                  />
                ) : (
                  <Camera size={16} aria-hidden="true" />
                )}
                {cameraStatus === "Starting" ? "Starting..." : "Start Camera"}
              </button>
              <button
                className="button"
                type="button"
                onClick={() => stopCamera()}
                disabled={cameraStatus !== "Active" && cameraStatus !== "Starting"}
              >
                Stop Camera
              </button>
            </div>
            {errorMessage ? (
              <p className="live-error" role="alert">
                {errorMessage}
              </p>
            ) : null}
          </Panel>

          <div className="stack">
            <Panel
              title="Detection status"
              description="Values update only when a response is received from the preprocessing API."
              icon={ScanFace}
              trailing={
                <StatusPill
                  tone={processingStatus === "Processing" ? "teal" : "neutral"}
                >
                  {processingStatus}
                </StatusPill>
              }
            >
              <div className="status-row">
                <span className="status-row-label">
                  <Activity size={15} className="nav-icon" aria-hidden="true" />
                  Faces detected
                </span>
                <strong className="status-value">
                  {response ? response.face_count : "—"}
                </strong>
              </div>
              <div className="status-row">
                <span className="status-row-label">
                  <Gauge size={15} className="nav-icon" aria-hidden="true" />
                  Image quality
                </span>
                <strong className="status-value">
                  {response ? `${response.quality_score.toFixed(1)} / 100` : "—"}
                </strong>
              </div>
              <div className="status-row">
                <span className="status-row-label">
                  <Activity size={15} className="nav-icon" aria-hidden="true" />
                  Blur variance
                </span>
                <strong className="status-value">
                  {response
                    ? response.quality_metrics.blur_variance.toFixed(1)
                    : "—"}
                </strong>
              </div>
              <div className="status-row">
                <span className="status-row-label">
                  {backendStatus === "Connected" ? (
                    <Wifi size={15} className="nav-icon" aria-hidden="true" />
                  ) : (
                    <WifiOff size={15} className="nav-icon" aria-hidden="true" />
                  )}
                  Backend
                </span>
                <StatusPill tone={backendTone}>{backendStatus}</StatusPill>
              </div>
              <div className="status-row">
                <span className="status-row-label">
                  <ShieldCheck size={15} className="nav-icon" aria-hidden="true" />
                  Camera session
                </span>
                <StatusPill tone={cameraTone}>{cameraStatus}</StatusPill>
              </div>
              {processingStatus === "Processing" ? (
                <p className="processing-note">
                  <LoaderCircle
                    size={14}
                    className="loading-icon"
                    aria-hidden="true"
                  />
                  Sending a captured frame to the local backend…
                </p>
              ) : null}
            </Panel>

            <Panel
              title="Quality warnings"
              description="Warnings describe image quality, not whether a face is genuine or spoofed."
              icon={Gauge}
            >
              {warnings.length > 0 ? (
                <div className="quality-warning-list">
                  {warnings.map((warning) => (
                    <span className="quality-warning" key={warning}>
                      {formatQualityFlag(warning)}
                    </span>
                  ))}
                </div>
              ) : response ? (
                <p className="quality-clear">No quality warnings reported.</p>
              ) : (
                <p className="muted-value">Waiting for a backend response.</p>
              )}
              {response && response.faces.length > 0 ? (
                <div className="face-result-list" aria-label="Detected faces">
                  {response.faces.map((face, index) => (
                    <article
                      className="face-result-card"
                      key={`${face.detection_index}-${face.face_id}`}
                    >
                      <div className="face-result-heading">
                        <strong>Face {index + 1}</strong>
                        <StatusPill>
                          Detection index {face.detection_index}
                        </StatusPill>
                      </div>
                      <div className="face-result-details">
                        <span>
                          Confidence:{" "}
                          {face.confidence === null
                            ? "Not provided"
                            : `${(face.confidence * 100).toFixed(1)}%`}
                        </span>
                        {face.quality_flags.length > 0 ? (
                          <div className="quality-warning-list">
                            {face.quality_flags.map((flag) => (
                              <span
                                className="quality-warning"
                                key={`${face.detection_index}-${flag}`}
                              >
                                {formatQualityFlag(flag)}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              ) : response ? (
                <p className="muted-value" style={{ marginTop: 12 }}>
                  No faces detected in the latest processed frame.
                </p>
              ) : null}
            </Panel>

            <Panel
              title="Classifier integration"
              description="This phase displays preprocessing results only."
              icon={ShieldCheck}
            >
              <div className="classification-placeholder">
                <div>
                  <CircleAlert
                    size={21}
                    className="notice-icon"
                    aria-hidden="true"
                  />
                  <strong>No classification result</strong>
                  <span>
                    No genuine/spoof label or confidence is generated.
                  </span>
                </div>
              </div>
            </Panel>
          </div>
        </div>

        <div className="notice" role="note">
          <ShieldCheck size={18} className="notice-icon" aria-hidden="true" />
          <p>
            Captured frames are sent only to the configured local preprocessing
            backend. Frames and biometric results are not saved to disk or
            browser storage.
          </p>
        </div>
      </div>
    </AppShell>
  );
}
