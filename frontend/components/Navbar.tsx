import Link from "next/link";

const links = [
  { href: "/predict", label: "Predict" },
  { href: "/history", label: "History" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  return (
    <header className="border-b border-clinical-border bg-clinical-surface">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2 text-clinical-ink">
          <svg
            viewBox="0 0 32 32"
            aria-hidden="true"
            className="h-7 w-7 shrink-0"
          >
            <rect width="32" height="32" rx="7" fill="#0a6cff" />
            <path d="M16 6 L16 11" stroke="#ffffff" strokeWidth="1.7" strokeLinecap="round" />
            <path d="M13.5 11 L18.5 11" stroke="#ffffff" strokeWidth="1.7" strokeLinecap="round" />
            <path
              d="M15 11 C 15 11, 8 11, 7 18 C 6.2 23, 8.4 24.5, 11 24.5 C 13.6 24.5, 15 23, 15 20 Z"
              fill="#ffffff"
            />
            <path
              d="M17 11 C 17 11, 24 11, 25 18 C 25.8 23, 23.6 24.5, 21 24.5 C 18.4 24.5, 17 23, 17 20 Z"
              fill="#ffffff"
            />
            <circle cx="10.5" cy="18" r="1.7" fill="#c1121f" />
            <circle
              cx="10.5"
              cy="18"
              r="3.2"
              fill="none"
              stroke="#c1121f"
              strokeWidth="0.7"
              strokeOpacity="0.6"
            />
          </svg>
          <span className="text-xl font-semibold tracking-tight">TB Detection</span>
          <span className="rounded bg-clinical-bg px-2 py-0.5 text-xs text-clinical-muted">
            DenseNet121 · Hybrid CNN+ViT
          </span>
        </Link>
        <nav className="flex gap-1 text-sm">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="rounded px-3 py-1.5 text-clinical-muted hover:bg-clinical-bg hover:text-clinical-ink"
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
