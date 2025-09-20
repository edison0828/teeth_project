import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "../components/Sidebar";
import TopNav from "../components/TopNav";

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
      <body className="flex bg-gradient-to-br from-[#020617] via-[#071026] to-[#0B1531]">
        <Sidebar />
        <div className="flex min-h-screen flex-1 flex-col">
          <TopNav />
          <main className="flex-1 overflow-y-auto p-8 lg:p-10 xl:p-12">{children}</main>
        </div>
      </body>
    </html>
  );
}
