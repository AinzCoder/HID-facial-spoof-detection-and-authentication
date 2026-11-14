"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import {
  Activity,
  Image as ImageIcon,
  Info,
  LayoutDashboard,
  Menu,
  ScanFace,
  ShieldCheck,
  X,
} from "lucide-react";

import { StatusPill } from "@/components/ui";

const navigation = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/live-detection", label: "Live Detection", icon: ScanFace },
  { href: "/image-analysis", label: "Image Analysis", icon: ImageIcon },
  { href: "/results", label: "Results", icon: Activity },
  { href: "/about", label: "About & System Info", icon: Info },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <div className="app-shell">
      {mobileNavOpen ? (
        <button
          className="mobile-backdrop"
          type="button"
          aria-label="Close navigation"
          onClick={() => setMobileNavOpen(false)}
        />
      ) : null}

      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <Link
          className="brand"
          href="/"
          aria-label="Sentinel dashboard"
          onClick={() => setMobileNavOpen(false)}
        >
          <span className="brand-mark">
            <ShieldCheck size={23} strokeWidth={1.8} aria-hidden="true" />
          </span>
          <span>
            <span className="brand-name">Sentinel</span>
            <span className="brand-caption">Biometric Security Lab</span>
          </span>
        </Link>

        <p className="nav-label">Workspace</p>
        <nav className="nav-list" aria-label="Main navigation">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                className={`nav-link ${active ? "nav-link-active" : ""}`}
                href={item.href}
                aria-current={active ? "page" : undefined}
                onClick={() => setMobileNavOpen(false)}
              >
                <Icon className="nav-icon" size={17} aria-hidden="true" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-spacer" />
        <section className="sidebar-status" aria-label="Backend status">
          <div className="sidebar-status-heading">
            <span className="inline-row">
              <Activity size={14} className="nav-icon" aria-hidden="true" />
              Service status
            </span>
            <StatusPill tone="amber">Not connected</StatusPill>
          </div>
          <p className="sidebar-status-copy">
            Backend health and classifier services are placeholders in this UI
            phase.
          </p>
        </section>
        <p className="brand-caption" style={{ padding: "16px 8px 0" }}>
          UNIVERSITY PROJECT · PHASE 1
        </p>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-button"
              type="button"
              aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"}
              aria-expanded={mobileNavOpen}
              onClick={() => setMobileNavOpen((open) => !open)}
            >
              {mobileNavOpen ? (
                <X size={18} aria-hidden="true" />
              ) : (
                <Menu size={18} aria-hidden="true" />
              )}
            </button>
            <span className="topbar-context">Security analysis workspace</span>
            <span className="topbar-divider" />
            <span className="topbar-caption">Preprocessing UI · Phase 1</span>
          </div>
          <div className="topbar-right">
            <StatusPill tone="amber">Backend not connected</StatusPill>
            <span className="topbar-caption">Local prototype</span>
          </div>
        </header>

        <main className="page-content">{children}</main>
        <footer className="page-footer">
          Sentinel · Face Anti-Spoofing Detection System · Research interface
        </footer>
      </div>
    </div>
  );
}
