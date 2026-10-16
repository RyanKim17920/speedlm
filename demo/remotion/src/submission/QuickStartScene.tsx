import React from "react";
import {DIM, MONO, PANEL, RULE, SERIES_GREEN, TEXT} from "../theme";
import {SubmissionFrame} from "./SubmissionFrame";

export const QuickStartScene: React.FC = () => (
  <SubmissionFrame
    eyebrow="QUICK START · KEEP YOUR EXISTING CLIENTS"
    title="Keep your app. Add SpeedLM to vLLM."
    provenance="README Quick Start · live-tested stack: Python 3.12 / vLLM 0.25.1 / CUDA 12.9 / H100"
    caption="Install once. Run speedlm doctor. Add --enable-idle-tuning. Your clients keep the same OpenAI-compatible API."
  >
    <div style={{width: "100%", display: "grid", gridTemplateColumns: "1.35fr 0.65fr", gap: 24}}>
      <section style={{background: "#f0f3f6", border: "1px solid #b7c0ca", borderRadius: 18, padding: "28px 32px"}}>
        <div style={{color: "#57606a", fontSize: 16, fontWeight: 800, marginBottom: 18}}>README COMMANDS · DOCUMENTATION CARD</div>
        <pre style={{margin: 0, color: "#1f2328", whiteSpace: "pre-wrap", font: `650 25px/1.5 ${MONO}`}}>
          <span style={{color: "#6e7781"}}>1  </span>git clone https://github.com/RyanKim17920/speedlm.git{"\n"}
          <span style={{color: "#6e7781"}}>2  </span>cd speedlm &amp;&amp; python -m pip install .{"\n"}
          <span style={{color: "#6e7781"}}>3  </span>speedlm doctor{"\n\n"}
          <span style={{color: "#1a7f37"}}><span style={{color: "#6e7781"}}>4  </span>speedlm vllm serve Qwen/Qwen3-8B \{"\n"}</span>
          <span style={{color: "#1a7f37"}}>      --enable-idle-tuning \{"\n"}</span>
          <span style={{color: "#1a7f37"}}>      --max-model-len 16384</span>
        </pre>
      </section>
      <section style={{display: "flex", flexDirection: "column", gap: 18}}>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${RULE}`, borderRadius: 18, padding: "25px 26px"}}>
          <div style={{color: SERIES_GREEN, fontSize: 21, fontWeight: 820}}>SET UP ONCE</div>
          <div style={{color: TEXT, fontSize: 22, lineHeight: 1.45, marginTop: 18}}>Choose the matching profile and point SpeedLM to your trainer and vLLM environments.</div>
        </div>
        <div style={{flex: 1, background: PANEL, border: `1px solid ${SERIES_GREEN}`, borderRadius: 18, padding: "25px 26px"}}>
          <div style={{color: SERIES_GREEN, fontSize: 21, fontWeight: 820}}>THEN KEEP SERVING</div>
          <div style={{color: TEXT, fontSize: 29, fontWeight: 800, marginTop: 18}}>one serving flag</div>
          <div style={{color: DIM, fontSize: 19, lineHeight: 1.4, marginTop: 12}}>Unconsumed vLLM arguments pass through.</div>
        </div>
      </section>
    </div>
  </SubmissionFrame>
);
