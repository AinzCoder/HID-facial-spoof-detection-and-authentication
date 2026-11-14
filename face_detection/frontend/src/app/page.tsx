import Link from "next/link";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Camera,
  Check,
  Cpu,
  Fingerprint,
  Image as ImageIcon,
  Layers,
  ScanFace,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import {
  MetricCard,
  PageHeading,
  Panel,
  StatusPill,
} from "@/components/ui";

const components = [
  {
    title: "Face detection",
    detail: "Local face localization and confidence metadata.",
    icon: ScanFace,
    status: "Implemented",
  },
  {
    title: "Landmarks & quality",
    detail: "Facial geometry and capture-quality indicators.",
    icon: Activity,
    status: "Implemented",
  },
  {
    title: "Crop & alignment",
    detail: "Context-preserving crops with optional alignment.",
    icon: ImageIcon,
    status: "Implemented",
  },
  {
    title: "Anti-spoof classifier",
    detail: "Team classifier contract is not yet verified.",
    icon: ShieldAlert,
    status: "Integration pending",
  },
];

export default function DashboardPage() {
  return (
    <AppShell>
      <div className="page-stack">
        <PageHeading
          eyebrow="Research workspace"
          title="Face Anti-Spoofing System"
          description="A modular workspace for evaluating face capture quality and preparing inputs for presentation-attack analysis."
          action={<StatusPill tone="amber">Phase 1 · UI prototype</StatusPill>}
        />

        <section className="hero-grid" aria-label="Project overview">
          <div className="hero-card">
            <p className="eyebrow">
              <span className="eyebrow-dot" />
              Capture integrity workspace
            </p>
            <h2 className="hero-title">
              Verify the signal.
              <br />
              <span className="hero-title-accent">Protect the session.</span>
            </h2>
            <p className="hero-copy">
              Explore a transparent face-processing workflow built for future
              anti-spoof classification. Preprocessing is implemented locally;
              classifier and API integration are not connected in this phase.
            </p>
            <div className="hero-actions">
              <Link className="button button-primary" href="/live-detection">
                Open live detection <ArrowRight size={16} aria-hidden="true" />
              </Link>
              <Link className="button" href="/image-analysis">
                Image analysis <ArrowUpRight size={15} aria-hidden="true" />
              </Link>
            </div>
          </div>
          <div className="scan-visual" aria-label="Abstract verification graphic">
            <div className="scan-grid" />
            <div className="scan-ring" />
            <div className="scan-ring scan-ring-inner" />
            <div className="scan-center">
              <Fingerprint size={39} strokeWidth={1.4} aria-hidden="true" />
            </div>
            <div className="scan-caption">
              <span>Interface preview</span>
              <span>No camera stream</span>
            </div>
          </div>
        </section>

        <section className="metric-grid" aria-label="System readiness">
          <MetricCard
            icon={ShieldCheck}
            label="Preprocessing"
            value="Implemented"
            detail="Local Python modules"
            tone="teal"
          />
          <MetricCard
            icon={Cpu}
            label="Classifier"
            value="Contract pending"
            detail="No results are simulated"
            tone="amber"
          />
          <MetricCard
            icon={Activity}
            label="Backend connection"
            value="Not connected"
            detail="API integration is future work"
            tone="neutral"
          />
          <MetricCard
            icon={Check}
            label="Privacy"
            value="User-controlled"
            detail="No camera or upload access"
            tone="teal"
          />
        </section>

        <Panel
          title="System components"
          description="Implemented preprocessing modules are distinct from the pending classifier integration."
          icon={Layers}
          trailing={<span className="topbar-caption">Local processing · no cloud service</span>}
        >
          <div className="component-grid">
            {components.map((component) => {
              const Icon = component.icon;
              return (
                <article className="component-card" key={component.title}>
                  <div className="component-card-top">
                    <Icon size={18} className="nav-icon" aria-hidden="true" />
                    <StatusPill
                      tone={
                        component.status === "Implemented" ? "teal" : "amber"
                      }
                    >
                      {component.status}
                    </StatusPill>
                  </div>
                  <div>
                    <h3 className="metric-value">{component.title}</h3>
                    <p>{component.detail}</p>
                  </div>
                </article>
              );
            })}
          </div>
        </Panel>

        <div className="notice" role="status">
          <Camera className="notice-icon" size={18} aria-hidden="true" />
          <p>
            Camera and image actions are intentionally inactive. The backend
            contract and classifier interface must be verified before data is
            captured or submitted.
          </p>
        </div>
      </div>
    </AppShell>
  );
}
