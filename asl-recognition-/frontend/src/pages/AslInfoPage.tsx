import { FiX } from "react-icons/fi";
import aslImage from "../../images/image.png";

export default function AslInfoPage({ onClose }) {
  return (
    <section className="glass-card info-page">
      <button className="close-icon info-close" type="button" onClick={onClose} aria-label="Close info page">
        <FiX />
      </button>
      <img className="info-full-image" src={aslImage} alt="ASL alphabet reference" />
    </section>
  );
}
