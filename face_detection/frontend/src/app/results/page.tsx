import { Activity, ArrowUpRight, Clock3 } from "lucide-react";
import Link from "next/link";

import { AppShell } from "@/components/app-shell";
import { EmptyState, PageHeading, Panel, StatusPill } from "@/components/ui";

export default function ResultsPage() {
  return (
    <AppShell>
      <div className="page-stack">
        <PageHeading
          eyebrow="Analysis records"
          title="Results"
          description="Review analysis results when a real preprocessing and classifier integration is available."
          action={<StatusPill>No records</StatusPill>}
        />

        <Panel
          title="Session results"
          description="This page does not create sample or placeholder detections."
          icon={Activity}
          trailing={<span className="topbar-caption">History is not stored</span>}
        >
          <EmptyState
            icon={Clock3}
            title="No analysis results yet"
            description="Run a real analysis after the image and live workflows are integrated. No detection history is generated or persisted by this prototype."
          />
        </Panel>

        <div className="notice" role="note">
          <Clock3 size={18} className="notice-icon" aria-hidden="true" />
          <p>
            Result history and retention behavior have not been designed. This
            UI does not store image data or create detection records.
          </p>
        </div>

        <div>
          <Link className="button" href="/">
            Return to dashboard <ArrowUpRight size={15} aria-hidden="true" />
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
