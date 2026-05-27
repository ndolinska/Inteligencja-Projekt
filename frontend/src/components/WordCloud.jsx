import { useState } from 'react'
import './WordCloud.css'

// Map frequency → font size range [minPx, maxPx]
function scaleSize(count, minCount, maxCount, minPx = 11, maxPx = 42) {
  if (maxCount === minCount) return (minPx + maxPx) / 2
  return minPx + ((count - minCount) / (maxCount - minCount)) * (maxPx - minPx)
}

const PALETTE = [
  '#6366f1', '#818cf8', '#a78bfa', '#c084fc',
  '#22c55e', '#34d399', '#06b6d4', '#38bdf8',
  '#f59e0b', '#fb923c', '#f472b6',
]

export default function WordCloud({ wordFrequency }) {
  const [hovered, setHovered] = useState(null)

  if (!wordFrequency?.length) return null

  const top = wordFrequency.slice(0, 60)
  const minCount = top[top.length - 1]?.count ?? 1
  const maxCount = top[0]?.count ?? 1

  return (
    <div className="chart-card word-cloud-card">
      <h3 className="chart-title">Najczęstsze słowa</h3>
      <div className="word-cloud">
        {top.map(({ word, count }, i) => {
          const size = scaleSize(count, minCount, maxCount)
          const color = PALETTE[i % PALETTE.length]
          const isHovered = hovered === word
          return (
            <span
              key={word}
              className="word-tag"
              style={{
                fontSize: `${size}px`,
                color: isHovered ? '#fff' : color,
                opacity: hovered && !isHovered ? 0.35 : 1,
                background: isHovered ? color : 'transparent',
              }}
              onMouseEnter={() => setHovered(word)}
              onMouseLeave={() => setHovered(null)}
              title={`${word}: ${count}×`}
            >
              {word}
            </span>
          )
        })}
      </div>
    </div>
  )
}
