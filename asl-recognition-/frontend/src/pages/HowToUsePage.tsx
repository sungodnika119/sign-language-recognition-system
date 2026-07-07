import { FiCheckCircle, FiVideo, FiX } from "react-icons/fi";

const steps = [
  "Choose Letters or Common Signs / Words from the recognition mode switch.",
  "Turn on the camera so the app can start reading your hand landmarks.",
  "Use Detect to watch live recognition, or Train to record a new sign sequence.",
  "In Detect mode, use Add Letter or Add Word to build text manually when needed.",
  "Use Space, Delete, Clear, Copy, Speak, and Generate Paragraph to finish the message.",
  "If you add a new word, save the 30-frame sequence and train the model from the Train tab.",
];

const tips = [
  "Keep your hand centered and well lit.",
  "Hold each sign steady for a moment so the model can settle.",
  "For words, the app works best with a smooth 30-frame sequence.",
];

export default function HowToUsePage({ onClose }) {
  return (
    <section className="glass-card howto-page">
      <button className="close-icon info-close" type="button" onClick={onClose} aria-label="Close how to use page">
        <FiX />
      </button>

      <div className="howto-hero">
        <div className="howto-head">
          <h2>How to Use the Web App</h2>
          <p>Simple steps to detect, type, speak, and train ASL signs.</p>
        </div>
        <div className="howto-banner">
          <FiVideo />
          <span>Camera on, sign steady, and the app will do the rest.</span>
        </div>
      </div>

      <div className="howto-group">
        {steps.map((step, index) => (
          <div className="howto-item" key={step}>
            <span className="howto-index">{index + 1}</span>
            <span className="howto-text">{step}</span>
            <FiCheckCircle className="howto-check" />
          </div>
        ))}
      </div>

      <div className="howto-group compact">
        {tips.map((tip) => (
          <div className="howto-tip" key={tip}>
            {tip}
          </div>
        ))}
      </div>
    </section>
  );
}
