import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VEIL — make machine perception measurable",
  description:
    "A research and testing platform for understanding how computer-vision systems perceive the physical world.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
