import { MetricCard } from './MetricCard.jsx';

const metricGroups = [
  {
    title: 'System',
    items: [
      { key: 'power', title: 'Energieverbrauch', valuePath: ['power_watts'], unit: 'W', icon: 'PWR', color: '#3b82f6' },
      { key: 'fps', title: 'Verarbeitungsrate', valuePath: ['fps'], unit: 'FPS', icon: 'FPS', color: '#10b981' },
    ],
  },
  {
    title: 'Spielererkennung',
    items: [
      { key: 'player-inference', title: 'Inferenzzeit', valuePath: ['models', 'player_detection', 'inference_ms'], unit: 'ms', icon: 'DET', color: '#f59e0b' },
      { key: 'player-count', title: 'Erkannte Spieler', valuePath: ['models', 'player_detection', 'count'], unit: 'Pers.', icon: 'CNT', color: '#8b5cf6' },
    ],
  },
  {
    title: 'Homographie (2D-Feld)',
    items: [
      { key: 'homography-inference', title: 'Inferenzzeit', valuePath: ['models', 'homography', 'inference_ms'], unit: 'ms', icon: 'HOM', color: '#f59e0b' },
      { key: 'homography-status', title: 'Matrizenstatus', valuePath: ['models', 'homography', 'status'], unit: '', icon: 'STA', color: '#10b981' },
    ],
  },
  {
    title: 'Lesbarkeit und OCR',
    items: [
      { key: 'readability-inference', title: 'Klassifikator', valuePath: ['models', 'readability', 'inference_ms'], unit: 'ms', icon: 'CLS', color: '#f59e0b' },
      { key: 'ocr-inference', title: 'OCR Oberkoerper', valuePath: ['models', 'ocr', 'inference_ms'], unit: 'ms', icon: 'OCR', color: '#f59e0b' },
    ],
  },
  {
    title: 'Formationsvorhersage',
    items: [
      { key: 'formation-inference', title: 'Inferenzzeit', valuePath: ['models', 'formation', 'inference_ms'], unit: 'ms', icon: 'XGB', color: '#f59e0b' },
      { key: 'formation-prediction', title: 'Vorhersage', valuePath: ['models', 'formation', 'prediction'], unit: '', icon: 'PRE', color: '#8b5cf6' },
      { key: 'formation-confidence', title: 'Vertrauen', valuePath: ['models', 'formation', 'confidence'], unit: '', icon: 'CONF', color: '#3b82f6', formatter: (value) => `${(value * 100).toFixed(1)}%` },
    ],
  },
];

function getNestedValue(source, path) {
  return path.reduce((accumulator, key) => accumulator?.[key], source);
}

export function MetricsSection({ telemetry }) {
  return (
    <section className="metrics-section">
      <h2 className="metrics-section__title">Echtzeit-Analyse der Submodelle</h2>

      <div className="metrics-section__grid">
        {metricGroups.map((group) => (
          <section key={group.title} className="metrics-group">
            <h3 className="metrics-group__title">{group.title}</h3>

            <div className="metrics-group__content">
              {group.items.map((item) => {
                const rawValue = getNestedValue(telemetry, item.valuePath);
                const displayValue = item.formatter ? item.formatter(rawValue) : rawValue;

                return (
                  <MetricCard
                    key={item.key}
                    title={item.title}
                    value={displayValue}
                    unit={item.unit}
                    icon={item.icon}
                    color={item.color}
                  />
                );
              })}

              {group.title === 'Lesbarkeit und OCR' ? (
                <div className="metrics-group__ocr-list">
                  <strong>Erkannte Rueckennummern:</strong> {telemetry.models?.ocr?.texts?.length > 0 ? telemetry.models.ocr.texts.join(', ') : 'Keine'}
                </div>
              ) : null}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}