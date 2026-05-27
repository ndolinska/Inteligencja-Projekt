import './CommentsSection.css'

function CommentCard({ comment, type }) {
  const score = comment.compound ?? comment.score ?? 0
  const formattedScore = typeof score === 'number'
    ? (score > 0 ? '+' : '') + score.toFixed(3)
    : null

  return (
    <div className={`comment-card comment-card--${type}`}>
      <div className="comment-header">
        <span className="comment-author">{comment.author ?? 'Anonimowy'}</span>
        {formattedScore && (
          <span className={`comment-score comment-score--${type}`}>
            {formattedScore}
          </span>
        )}
        {comment.like_count > 0 && (
          <span className="comment-likes">♥ {comment.like_count.toLocaleString('pl-PL')}</span>
        )}
      </div>
      <p className="comment-text">{comment.text}</p>
    </div>
  )
}

function Column({ title, comments, type, accent }) {
  if (!comments?.length) return null
  return (
    <div className="comments-col">
      <h3 className="comments-col-title" style={{ color: accent }}>
        {title}
      </h3>
      <div className="comments-list">
        {comments.map((c, i) => (
          <CommentCard key={i} comment={c} type={type} />
        ))}
      </div>
    </div>
  )
}

export default function CommentsSection({ topPositive, topNegative }) {
  return (
    <div className="chart-card comments-section">
      <h3 className="chart-title">Przykładowe komentarze</h3>
      <div className="comments-grid">
        <Column
          title="▲ Najbardziej pozytywne"
          comments={topPositive}
          type="positive"
          accent="#22c55e"
        />
        <Column
          title="▼ Najbardziej negatywne"
          comments={topNegative}
          type="negative"
          accent="#ef4444"
        />
      </div>
    </div>
  )
}
