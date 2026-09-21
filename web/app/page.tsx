import Link from "next/link";

/**
 * Landing page. Every claim here must be one the platform can back with a
 * measurement - see docs/product.md. No invisibility language.
 */

const SECTIONS = [
  {
    title: "How it works",
    body: "You upload imagery you are authorized to test with, pick an open-source detector, and define a transformation distribution — angles, scales, lighting, camera blur, cloth deformation. VEIL measures the detector's behaviour across that distribution, optimizes a candidate pattern against it, and measures again.",
  },
  {
    title: "VEIL Lab",
    body: "The testing engine. Experiments are reproducible by construction: every run pins its seed, code version, detector weights, dataset version and full configuration. Re-run it and you get the same numbers and the same pattern file.",
  },
  {
    title: "VEIL Wear",
    body: "Physical garments carrying tested patterns. Each item records its pattern ID, batch, material, print method and the conditions it was measured under — resolvable from a tag back to the experiment that produced it.",
  },
  {
    title: "Physical testing",
    body: "A printed pattern, a real camera, a recorded distance, angle, lighting and environment. Physical results live in their own table, their own score and their own section of every report. They are never merged with simulated ones.",
  },
  {
    title: "Research methodology",
    body: "Patterns are optimized over an expectation across transformations, not against a single static image, and are constrained to a printable ink palette. Raw per-sample records are stored, so every aggregate is recomputable without re-running the model.",
  },
  {
    title: "Reports",
    body: "JSON and PDF. Detector version, dataset provenance, transformation grid, baseline and candidate rates with 95% intervals, per-axis breakdown, physical tests, and a limitations section that is not optional.",
  },
];

export default function Landing() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-20 md:py-28">
      <header>
        <p className="text-sm tracking-[0.4em]" style={{ color: "var(--ink-muted)" }}>VEIL</p>
        <h1 className="mt-8 text-4xl leading-tight md:text-5xl">Make machine perception measurable.</h1>
        <p className="mt-6 max-w-2xl text-base leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
          A research and testing platform for understanding how computer-vision systems
          perceive the physical world.
        </p>
        <div className="mt-10 flex flex-wrap gap-3 text-sm">
          <Link href="/dashboard" className="rounded border px-5 py-3" style={{ borderColor: "var(--border)" }}>
            Open VEIL Lab →
          </Link>
          <a href="#waitlist" className="rounded px-5 py-3" style={{ background: "var(--surface-raised)" }}>
            Join the waitlist
          </a>
        </div>
      </header>

      <div className="mt-24 space-y-14">
        {SECTIONS.map((section) => (
          <section key={section.title}>
            <h2 className="text-xs uppercase tracking-[0.2em]" style={{ color: "var(--ink-muted)" }}>
              {section.title}
            </h2>
            <p className="mt-3 leading-relaxed" style={{ color: "var(--ink-secondary)" }}>{section.body}</p>
          </section>
        ))}

        <section className="rounded-lg border p-6" style={{ borderColor: "var(--border)" }}>
          <h2 className="text-xs uppercase tracking-[0.2em]" style={{ color: "var(--ink-muted)" }}>
            Limitations
          </h2>
          <ul className="mt-3 space-y-2 text-sm leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
            <li>
              VEIL measures detector behaviour under stated conditions. It does not provide, and
              does not claim, invisibility from surveillance systems.
            </li>
            <li>
              A result is specific to the model, dataset, camera and environment it was measured
              on. It is not expected to transfer to systems that were not measured.
            </li>
            <li>
              Simulated results approximate physical capture. They are not a substitute for it.
            </li>
            <li>Detection rates are estimates from finite samples and are reported with intervals.</li>
          </ul>
        </section>

        <section id="waitlist">
          <h2 className="text-xs uppercase tracking-[0.2em]" style={{ color: "var(--ink-muted)" }}>Waitlist</h2>
          <p className="mt-3 text-sm leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
            VEIL is pre-release. For research access or a robustness evaluation, write to{" "}
            <span style={{ color: "var(--ink)" }}>mohamed-aziz.ghedamsi@epitech.eu</span>.
          </p>
        </section>
      </div>
    </div>
  );
}
