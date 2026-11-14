import {
  ArrowRight,
  FileCheck,
  Image as ImageIcon,
  ScanFace,
  ShieldCheck,
  Upload,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { PageHeading, Panel, StatusPill } from "@/components/ui";

export default function ImageAnalysisPage() {
  return (
    <AppShell>
      <div className="page-stack">
        <PageHeading
          eyebrow="Single-image workspace"
          title="Image Analysis"
          description="Prepare an image for future face preprocessing and anti-spoof analysis. Uploads and backend requests are not enabled in Phase 1."
          action={<StatusPill tone="amber">Integration pending</StatusPill>}
        />

        <div className="split-grid">
          <Panel
            title="Image input"
            description="The upload target is a visual placeholder and does not accept or retain files."
            icon={ImageIcon}
            trailing={<StatusPill>JPG · PNG</StatusPill>}
          >
            <div className="dropzone" aria-label="Image upload unavailable">
              <div>
                <span className="dropzone-icon">
                  <Upload size={24} aria-hidden="true" />
                </span>
                <h2>Image upload is not active</h2>
                <p>
                  This panel will support a user-selected image after API
                  validation, data handling, and privacy requirements are
                  agreed.
                </p>
              </div>
            </div>
            <div className="inline-row" style={{ marginTop: 16 }}>
              <button className="button button-primary" type="button" disabled>
                <Upload size={15} aria-hidden="true" /> Choose image
              </button>
              <button className="button" type="button" disabled>
                <ArrowRight size={15} aria-hidden="true" /> Analyze image
              </button>
            </div>
          </Panel>

          <div className="stack">
            <Panel
              title="Analysis result"
              description="Results will appear after the preprocessing API is connected."
              icon={FileCheck}
            >
              <div className="classification-placeholder">
                <div>
                  <ScanFace
                    size={23}
                    className="nav-icon"
                    aria-hidden="true"
                  />
                  <strong>Nothing to analyze yet</strong>
                  <span>
                    No face count, quality score, or classifier result is
                    fabricated.
                  </span>
                </div>
              </div>
            </Panel>

            <Panel
              title="Data handling"
              description="Privacy controls for the future upload workflow."
              icon={ShieldCheck}
            >
              <div className="status-row">
                <span className="status-row-label">Automatic upload</span>
                <span className="muted-value">Disabled</span>
              </div>
              <div className="status-row">
                <span className="status-row-label">Automatic storage</span>
                <span className="muted-value">Disabled</span>
              </div>
              <div className="status-row">
                <span className="status-row-label">Backend API</span>
                <StatusPill tone="amber">Not connected</StatusPill>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
