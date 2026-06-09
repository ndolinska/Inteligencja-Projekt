import StatsRow from './StatsRow'
import SentimentDonut from './SentimentDonut'
import TimeSeries from './TimeSeries'
import LanguageChart from './LanguageChart'
import WordCloud from './WordCloud'
import CommentsSection from './CommentsSection'
import TranscriptPanel from './TranscriptPanel'
import './Dashboard.css'

export default function Dashboard({ data, videoUrl }) {
  const videoId = data.video_id
  const thumbUrl = videoId
    ? `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`
    : null

  return (
    <div className="dashboard">
      {/* Video info banner */}
      <div className="video-banner">
        {thumbUrl && (
          <img
            className="video-thumb"
            src={thumbUrl}
            alt="miniatura wideo"
          />
        )}
        <div className="video-meta">
          <span className="video-label">Analizowane wideo</span>
          <a
            href={videoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="video-url"
          >
            {videoUrl}
          </a>
        </div>
      </div>

      {/* Summary stats */}
      <StatsRow data={data} />

      {/* Charts grid */}
      <div className="charts-grid">
        <SentimentDonut distribution={data.distribution} />
        <TimeSeries timeSeries={data.time_series} />
        <LanguageChart languages={data.languages} />
        <WordCloud wordFrequency={data.word_frequency} />
        <CommentsSection
          topPositive={data.top_positive}
          topNegative={data.top_negative}
        />
        <TranscriptPanel
          transcript={data.transcript_sentiment}
          comments={{ distribution: data.distribution, percentages: data.percentages }}
        />
      </div>
    </div>
  )
}
