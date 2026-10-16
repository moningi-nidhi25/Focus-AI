import {
  Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { CurvePoint, TrendPoint } from "../dashApi";

const AXIS = "#64748b";
const GRID = "rgba(148, 163, 184, 0.2)";

const tip = {
  contentStyle: {
    background: "#0f172a",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: 10,
    color: "#f8fafc",
    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3)",
    fontSize: 12,
    padding: "8px 12px",
  },
  itemStyle: {
    padding: "2px 0",
    color: "#f1f5f9",
  },
  cursor: {
    stroke: "#94a3b8",
    strokeWidth: 1,
    strokeDasharray: "3 3",
  },
};

const margin = { top: 12, right: 16, left: -14, bottom: 4 };
const shortDate = (d: string) => d.slice(5);

export function StatCard({ label, value, hint }: { label: string; value: string | number | null; hint?: string }) {
  const isUp = hint?.includes("▲");
  const isDown = hint?.includes("▼");
  const trendClass = isUp ? "trend-up" : isDown ? "trend-down" : "";

  return (
    <div className="d-card stat">
      <div className="d-label">{label}</div>
      <div className="d-value">{value ?? "–"}</div>
      {hint && <div className={`d-hint ${trendClass}`}>{hint}</div>}
    </div>
  );
}

export function Insights({ items }: { items: string[] }) {
  if (!items || items.length === 0) return null;

  return (
    <div className="d-card">
      <div className="d-card-header">
        <h3>Personalized insights</h3>
        <p className="d-hint">Actionable takeaways based on your recent activity patterns.</p>
      </div>
      <ul className="d-list">
        {items.map((t, idx) => (
          <li key={idx}>{t}</li>
        ))}
      </ul>
    </div>
  );
}

export function DaysSelect({ value, onChange }: { value: number; onChange: (d: number) => void }) {
  return (
    <div className="d-select-wrapper">
      <select
        className="d-select"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label="Select timeframe"
      >
        {[7, 14, 30, 90].map((d) => (
          <option key={d} value={d}>
            Last {d} days
          </option>
        ))}
      </select>
      <svg className="d-select-arrow" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
    </div>
  );
}

export function FocusTrend({ data }: { data: TrendPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={margin}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} stroke={AXIS} fontSize={11} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis domain={[0, 100]} stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip {...tip} />
        <Line
          type="monotone"
          dataKey="avg_focus"
          name="Avg focus"
          stroke="#4f46e5"
          strokeWidth={2.5}
          dot={{ r: 3.5, fill: "#4f46e5", stroke: "#ffffff", strokeWidth: 1.5 }}
          activeDot={{ r: 6, fill: "#4338ca", stroke: "#ffffff", strokeWidth: 2 }}
          connectNulls={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function MinutesChart({ data }: { data: TrendPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} margin={margin}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} stroke={AXIS} fontSize={11} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip {...tip} />
        <Bar dataKey="study_minutes" name="Study minutes" fill="#10b981" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DistractionChart({ data }: { data: TrendPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} margin={margin}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="date" tickFormatter={shortDate} stroke={AXIS} fontSize={11} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} allowDecimals={false} />
        <Tooltip {...tip} />
        <Bar dataKey="distractions" name="Distractions" fill="#ef4444" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DistributionChart({ data }: { data: { range: string; students: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} margin={margin}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="range" stroke={AXIS} fontSize={11} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis allowDecimals={false} stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip {...tip} />
        <Bar dataKey="students" name="Students" fill="#6366f1" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DecayChart({ current, curve }: { current?: number | null; curve: CurvePoint[] }) {
  const points = curve.map((c) => ({ window: c.window, predicted: c.predicted, band: [c.lower, c.upper] }));
  const data = current == null ? points : [{ window: 0, predicted: current, band: [current, current] }, ...points];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={margin}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="window"
          stroke={AXIS}
          fontSize={11}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          tickFormatter={(w: number) => (w === 0 ? "Now" : `+${w}`)}
        />
        <YAxis domain={[0, 100]} stroke={AXIS} fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip {...tip} />
        <ReferenceLine
          y={50}
          stroke="#f59e0b"
          strokeDasharray="4 4"
          strokeWidth={1.5}
          label={{ value: "Alert threshold (50)", fill: "#f59e0b", fontSize: 11, position: "insideTopRight" }}
        />
        <Area
          type="monotone"
          dataKey="band"
          name="Likely range"
          stroke="none"
          fill="#4f46e5"
          fillOpacity={0.16}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          name="Predicted focus"
          stroke="#4f46e5"
          strokeWidth={2.5}
          dot={{ r: 3.5, fill: "#4f46e5", stroke: "#ffffff", strokeWidth: 1.5 }}
          activeDot={{ r: 6, fill: "#4338ca", stroke: "#ffffff", strokeWidth: 2 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}