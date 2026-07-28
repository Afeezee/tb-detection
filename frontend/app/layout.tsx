import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "TB Detection — Deep Learning Chest X-ray Screening",
  description:
    "Chest X-ray screening for pulmonary tuberculosis using DenseNet121 and a Hybrid CNN+ViT model.",
  applicationName: "TB Detection",
  manifest: "/site.webmanifest",
  themeColor: "#0a6cff",
  icons: {
    icon: [
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    shortcut: [{ url: "/icon.svg", type: "image/svg+xml" }],
    apple: [{ url: "/apple-icon.svg", type: "image/svg+xml", sizes: "180x180" }],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en-GB">
      <body>
        <Navbar />
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-6 py-8 text-xs text-clinical-muted">
          TB Detection Research Project · Adesanlu Martins (U/22/CS/0011) · Supervisor: Miss Shadare
          <br />
          Research prototype only — not a licensed medical device.
        </footer>
      </body>
    </html>
  );
}
