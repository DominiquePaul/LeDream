import type { Metadata } from "next";
import { DM_Mono, Geist } from "next/font/google";
import "./globals.css";
import Shell from "@/components/Shell";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const dmMono = DM_Mono({
  variable: "--font-dm-mono",
  weight: ["400", "500"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DreamHub",
  description: "Robotics experiment visualization and management hub — Dream Machines",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${dmMono.variable} antialiased bg-gray-950`}
      >
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
