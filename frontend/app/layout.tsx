import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "SEM Fiber Analysis", description: "Human-in-the-loop fiber thickness review" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body>{children}</body></html>;
}
