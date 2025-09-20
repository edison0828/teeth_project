interface ProgressCircleProps {
  value: number;
  label: string;
}

export default function ProgressCircle({ value, label }: ProgressCircleProps) {
  const displayValue = Math.round(value * 100);
  const offset = 283 - (283 * displayValue) / 100;

  return (
    <div className="relative h-32 w-32">
      <svg className="h-full w-full" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" stroke="rgba(148,163,184,0.2)" strokeWidth="8" fill="transparent" />
        <circle
          cx="50"
          cy="50"
          r="45"
          stroke="url(#gradient)"
          strokeWidth="8"
          strokeDasharray="283"
          strokeDashoffset={offset}
          strokeLinecap="round"
          fill="transparent"
        />
        <defs>
          <linearGradient id="gradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#3CC4FF" />
            <stop offset="100%" stopColor="#26E1A6" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-semibold text-white">{displayValue}%</span>
        <span className="mt-1 text-xs uppercase tracking-wide text-slate-400">{label}</span>
      </div>
    </div>
  );
}
