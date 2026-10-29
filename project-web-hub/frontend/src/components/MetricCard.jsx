export function MetricCard({ title, value, unit, icon, color }) {
  return (
    <article className="metric-card" style={{ borderLeftColor: color }}>
      <div className="metric-card__header">
        {icon ? (
          <span className="metric-card__icon" style={{ color, fontWeight: 'bold', fontSize: '0.8rem', marginRight: '6px', fontFamily: 'monospace' }}>
            [{icon}]
          </span>
        ) : null}
        <h4 className="metric-card__title">{title}</h4>
      </div>
      <div className="metric-card__body">
        <span className="metric-card__value" style={{ color }}>
          {value}
        </span>
        {unit ? <span className="metric-card__unit">{unit}</span> : null}
      </div>
    </article>
  );
}