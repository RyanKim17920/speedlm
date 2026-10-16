import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT } from "../theme";
import { AUTHENTIC_CAPTURE } from "./authenticCapture";

const commands = {
  normal: [
    "vllm serve Qwen/Qwen3-8B \\",
    "  --max-model-len 24576 \\",
    "  --gpu-memory-utilization 0.85 \\",
    "  --enforce-eager",
  ],
  speedlm: [
    "speedlm vllm serve Qwen/Qwen3-8B \\",
    "  --config capture-config.json \\",
    "  --enable-idle-tuning \\",
    "  --max-model-len 24576 \\",
    "  --gpu-memory-utilization 0.85 \\",
    "  --enforce-eager",
  ],
};

const Panel: React.FC<{
  accent: string;
  command: readonly string[];
  label: string;
  frame: number;
  startupSeconds: number;
}> = ({ accent, command, label, frame, startupSeconds }) => {
  const showReady = frame >= 88;
  const showRequest = frame >= 150;
  const responseChars = Math.floor(interpolate(frame, [178, 225], [0, AUTHENTIC_CAPTURE.response.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.linear,
  }));

  return (
    <section style={{ flex: 1, height: 800, border: `1px solid ${RULE}`, borderRadius: 22, background: PANEL, overflow: "hidden" }}>
      <div style={{ height: 78, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 26px", borderBottom: `1px solid ${RULE}` }}>
        <div style={{ color: accent, fontSize: 31, fontWeight: 760 }}>{label}</div>
        <div style={{ color: showReady ? SERIES_GREEN : DIM, fontSize: 19, fontWeight: 700 }}>{showReady ? "READY" : "STARTING"}</div>
      </div>
      <div style={{ padding: "28px 30px", color: TEXT }}>
        <div style={{ color: DIM, fontSize: 18, marginBottom: 14 }}>COMMAND</div>
        <pre style={{ margin: 0, whiteSpace: "pre-wrap", color: TEXT, font: `500 23px/1.55 ${MONO}` }}>{command.join("\n")}</pre>

        <div style={{ height: 1, background: RULE, margin: "28px 0" }} />
        <div style={{ fontSize: 22, lineHeight: 1.7 }}>
          <div style={{ color: showReady ? SERIES_GREEN : DIM }}>
            {showReady ? `✓ API ready · real startup ${startupSeconds.toFixed(1)}s` : "Loading Qwen3-8B…"}
          </div>
          {showRequest ? (
            <>
              <div style={{ color: DIM, marginTop: 24, fontSize: 18 }}>POST /v1/chat/completions</div>
              <div style={{ color: TEXT, marginTop: 8 }}>{AUTHENTIC_CAPTURE.request}</div>
              <div style={{ color: DIM, marginTop: 24, fontSize: 18 }}>RESPONSE</div>
              <div style={{ color: accent, marginTop: 8, minHeight: 42 }}>
                {AUTHENTIC_CAPTURE.response.slice(0, responseChars)}
                {responseChars < AUTHENTIC_CAPTURE.response.length ? "▌" : ""}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </section>
  );
};

export const AuthenticServeScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: BG, padding: "54px 80px", fontFamily: MONO }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 28 }}>
        <div style={{ color: TEXT, fontSize: 50, fontWeight: 740 }}>Run either server normally.</div>
        <div style={{ color: DIM, fontSize: 19 }}>fresh two-H100 capture · job {AUTHENTIC_CAPTURE.jobId}</div>
      </div>
      <div style={{ display: "flex", gap: 34 }}>
        <Panel accent={SERIES_BLUE} command={commands.normal} label="Normal vLLM" frame={frame} startupSeconds={AUTHENTIC_CAPTURE.normalStartupSeconds} />
        <Panel accent={SERIES_GREEN} command={commands.speedlm} label="SpeedLM vLLM" frame={frame} startupSeconds={AUTHENTIC_CAPTURE.speedlmStartupSeconds} />
      </div>
      {frame >= 228 ? (
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 28, color: SERIES_GREEN, textAlign: "center", fontSize: 24, fontWeight: 760 }}>
          IDENTICAL OUTPUT · {AUTHENTIC_CAPTURE.promptTokens} input tokens · {AUTHENTIC_CAPTURE.completionTokens} output tokens
        </div>
      ) : null}
    </AbsoluteFill>
  );
};
