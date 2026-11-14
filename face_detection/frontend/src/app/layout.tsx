import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Sentinel | Face Anti-Spoofing System",
    template: "%s | Sentinel",
  },
  description:
    "A university research interface for face preprocessing and future anti-spoof analysis.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
