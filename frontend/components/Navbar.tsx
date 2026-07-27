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
