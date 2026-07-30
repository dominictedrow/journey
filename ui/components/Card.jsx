export default function Card({ title, meta, children }) {
  return (
    <div className="card">
      <div className="card-title-row">
        <h3>{title}</h3>
        {meta && <span className="meta">{meta}</span>}
      </div>
      {children}
    </div>
  );
}
