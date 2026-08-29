import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FiberVision",
  description: "SEM fiber thickness analysis and human review workspace",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body>{children}</body></html>;
}
