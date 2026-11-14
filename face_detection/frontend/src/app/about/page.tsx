import {
  Activity,
  Cpu,
  Eye,
  Fingerprint,
  Image as ImageIcon,
  LockKeyhole,
  ScanFace,
  ShieldAlert,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { PageHeading, StatusPill } from "@/components/ui";

const information = [
  {
    title: "Face detection",
    icon: ScanFace,
    text: "Locates one or more faces in an image and provides bounding boxes and detector confidence where available. Detection identifies a face region; it does not identify a person.",
  },
  {
    title: "Facial landmark extraction",
    icon: Fingerprint,
    text: "Extracts facial geometry used by downstream preprocessing, including normalized and pixel-space landmark coordinates. Landmarks describe estimated geometry and are not identity data by themselves.",
  },
  {
    title: "Image quality validation",
    icon: Eye,
    text: "Measures image resolution, blur, brightness, face size, boundary clipping, and approximate pose. The thresholds are heuristics and require calibration for the intended capture conditions; landmarks cannot guarantee visibility under occlusion.",
  },
  {
    title: "Face cropping and alignment",
    icon: ImageIcon,
    text: "Produces context-preserving face crops and supports optional geometric alignment. Alignment is disabled by default; original crops remain available to downstream models.",
  },
  {
    title: "Anti-spoof classification",
    icon: ShieldAlert,
    text: "The classifier and its input/output contract are not verified or connected in this frontend phase. This interface does not predict genuine/spoof labels or confidence values.",
  },
  {
    title: "Purpose and privacy",
    icon: LockKeyhole,
    text: "The project is a university research system for exploring face capture preprocessing and future presentation-attack detection. Browser camera access, file upload, backend calls, and automatic storage are not active in this UI prototype.",
  },
];

export default function AboutPage() {
  return (
    <AppShell>
      <div className="page-stack">
        <PageHeading
          eyebrow="Project documentation"
          title="About the system"
          description="An overview of the preprocessing pipeline, intended classifier integration, and current frontend limitations."
          action={<StatusPill tone="teal">Preprocessing implemented</StatusPill>}
        />

        <div className="notice" role="note">
          <Activity size={18} className="notice-icon" aria-hidden="true" />
          <p>
            <strong>Implementation boundary:</strong> face preprocessing is
            implemented in the Python backend. Anti-spoof classifier
            availability and its API contract remain unverified. The frontend
            is visual only and does not invoke either component.
          </p>
        </div>

        <section className="info-grid" aria-label="System components and purpose">
          {information.map((item) => {
            const Icon = item.icon;
            return (
              <article className="info-card" key={item.title}>
                <h2>
                  <Icon size={18} className="nav-icon" aria-hidden="true" />
                  {item.title}
                  {item.title === "Anti-spoof classification" ? (
                    <StatusPill tone="amber">Unverified</StatusPill>
                  ) : null}
                </h2>
                <p>{item.text}</p>
              </article>
            );
          })}
        </section>

        <div className="metric-grid">
          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-icon">
                <Cpu size={17} aria-hidden="true" />
              </span>
              <StatusPill tone="teal">Local</StatusPill>
            </div>
            <p className="metric-value">Python preprocessing</p>
            <p className="metric-label">MediaPipe and OpenCV modules</p>
          </div>
          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-icon">
                <LockKeyhole size={17} aria-hidden="true" />
              </span>
              <StatusPill>Not active</StatusPill>
            </div>
            <p className="metric-value">Web and API integration</p>
            <p className="metric-label">No camera, uploads, or network calls</p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
