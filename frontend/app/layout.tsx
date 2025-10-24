import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DentaMind AI Platform",
  description: "Unified workspace for oral X-ray analytics and patient care"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-[#020617] via-[#071026] to-[#0B1531] text-slate-100">
        {children}
      </body>
    </html>
  );
}
