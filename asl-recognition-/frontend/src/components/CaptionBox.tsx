export default function CaptionBox({ caption, isSpeaking }) {
  const displayText = caption && caption.trim() ? caption : "Captions will appear here...";
  return (
    <section className={`caption-box ${isSpeaking ? "speaking" : ""}`}>
      <span className="caption-label">{isSpeaking ? "Speaking..." : "Live Caption"}</span>
      <div className="caption-text">{displayText}</div>
    </section>
  );
}
