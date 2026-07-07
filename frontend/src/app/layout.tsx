import type { Metadata } from "next";

import { Nav } from "@/components/layout/Nav";

import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Copytrader",
  description: "Polymarket intelligence & copy-trading dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-900 text-slate-100 antialiased">
        <Providers>
          <div className="flex min-h-screen">
            <Nav />
            <main className="flex-1 p-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
