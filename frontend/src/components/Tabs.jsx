import { useMemo, useState } from 'react';
import './Tabs.css';

function Tabs({ tabs, render }) {
  const firstId = tabs?.[0]?.id || null;
  const [activeId, setActiveId] = useState(firstId);

  const active = useMemo(() => tabs.find((t) => t.id === activeId) || tabs[0], [tabs, activeId]);
  if (!tabs?.length) return null;

  return (
    <div className="tabs">
      <div className="tab-list">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tab ${t.id === activeId ? 'active' : ''}`}
            onClick={() => setActiveId(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="tab-content">{render(active)}</div>
    </div>
  );
}

export default Tabs;

