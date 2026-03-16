import type { Metadata } from "next";
import "./globals.css";
import { NextAuthProvider } from "@/providers/NextAuthProvider";

export const metadata: Metadata = {
  title: "Market Sentiment Predictor",
  description: "Predict tomorrow's market direction from ticker sentiment.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <NextAuthProvider>{children}</NextAuthProvider>
      </body>
    </html>
  );
}
