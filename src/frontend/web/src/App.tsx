import { useState } from "react";
import "./App.css";
import Generate from "./pages/Generate";
import Monitor from "./pages/Monitor";

type Tab = "generate" | "monitor";

function App() {
  const [tab, setTab] = useState<Tab>("generate");

  return (
    <div className="page">
      <header>
        <h1>Text-to-Image — DDPM vs Flow Matching</h1>
        <p className="subtitle">
          Génération d'images de visages conditionnée par texte, entraînée from scratch sur CelebA-Dialog.
        </p>
      </header>

      <nav className="tabs">
        <button className={tab === "generate" ? "tab active" : "tab"} onClick={() => setTab("generate")}>
          Générer
        </button>
        <button className={tab === "monitor" ? "tab active" : "tab"} onClick={() => setTab("monitor")}>
          Suivi entraînement
        </button>
      </nav>

      {tab === "generate" ? <Generate /> : <Monitor />}
    </div>
  );
}

export default App;
