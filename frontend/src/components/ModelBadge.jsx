import './ModelBadge.css';

function ModelBadge({ text }) {
  if (!text) return null;
  return <span className="badge">{text}</span>;
}

export default ModelBadge;

