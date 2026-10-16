import type { Sentence } from "../api";

function shade(score: number): string {
  if (score >= 60) return `rgba(34, 197, 94, ${0.2 + (0.45 * (score - 60)) / 40})`;
  if (score >= 30) return `rgba(250, 204, 21, ${0.22 + (0.3 * (score - 30)) / 30})`;
  return `rgba(239, 68, 68, ${0.06 + (0.1 * (30 - score)) / 30})`;
}

export default function Heatmap({ sentences, method }: { sentences: Sentence[]; method: string }) {
  if (sentences.length === 0) return <p className="muted">No readable text to highlight.</p>;

  return (
    <div className="heatmap-container">
      <p className="heatmap" aria-label="Relevance heatmap">
        {sentences.map((s, i) => (
          <span
            key={i}
            className="sentence"
            style={{ backgroundColor: shade(s.score) }}
            title={`Sentence relevance: ${Math.round(s.score)}%`}
          >
            {s.text}{" "}
          </span>
        ))}
      </p>
      <div className="legend">
        <span><i style={{ background: shade(80) }} /> Relevant (≥60%)</span>
        <span><i style={{ background: shade(45) }} /> Partially relevant (30–59%)</span>
        <span><i style={{ background: shade(5) }} /> Unrelated (&lt;30%)</span>
        <span className="muted" style={{ marginLeft: "auto" }}>Method: {method}</span>
      </div>
    </div>
  );
}
