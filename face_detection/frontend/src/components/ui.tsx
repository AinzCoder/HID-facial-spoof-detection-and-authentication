import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

type StatusTone = "neutral" | "teal" | "amber";

export function StatusPill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: StatusTone;
}) {
  return (
    <span className={`status-pill status-pill-${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      {children}
    </span>
  );
}

export function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-heading">
      <div>
        <p className="eyebrow">
          <span className="eyebrow-dot" />
          {eyebrow}
        </p>
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action ? <div>{action}</div> : null}
    </header>
  );
}

export function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral",
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  detail: string;
  tone?: StatusTone;
}) {
  return (
    <article className="metric-card">
      <div className="metric-top">
        <span className="metric-icon">
          <Icon size={17} aria-hidden="true" />
        </span>
        <StatusPill tone={tone}>{label}</StatusPill>
      </div>
      <p className="metric-value">{value}</p>
      <p className="metric-label">{detail}</p>
    </article>
  );
}

export function Panel({
  title,
  description,
  icon: Icon,
  trailing,
  children,
}: {
  title: string;
  description?: string;
  icon?: LucideIcon;
  trailing?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div className="panel-title-wrap">
          {Icon ? (
            <span className="panel-icon">
              <Icon size={18} aria-hidden="true" />
            </span>
          ) : null}
          <div>
            <h2 className="panel-title">{title}</h2>
            {description ? (
              <p className="panel-description">{description}</p>
            ) : null}
          </div>
        </div>
        {trailing}
      </div>
      {children}
    </section>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
}) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon">
        <Icon size={27} strokeWidth={1.6} aria-hidden="true" />
      </span>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}
