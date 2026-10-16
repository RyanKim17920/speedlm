import React from "react";
import {Video} from "@remotion/media";
import {AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";

const LAST_REQUEST_CAST_SECONDS = 139.614822;
const IDLE_DETECTED_CAST_SECONDS = 146.516406;
const VLLM_ASLEEP_CAST_SECONDS = 155.728715;
const TRAINING_STARTED_CAST_SECONDS = 358.989296;
const SPEEDLM_TRIM_FRAMES = 339;

// Exact cast -> MP4 time mapping from pty-speedlm-server-timing.json.  Gaps in
// the real recording were compressed, so the displayed clock must be derived
// from these timestamps instead of pretending the edit runs in real time.
const CAST_TO_VIDEO: ReadonlyArray<readonly [number, number]> = [
  [124.320845, 8.189343],
  [139.614822, 12.012837],
  [142.033301, 12.512837],
  [145.905039, 13.012837],
  [146.546665, 13.173244],
  [155.668054, 13.673244],
  [155.969627, 13.748637],
  [159.401104, 14.248637],
  [358.989296, 14.748637],
] as const;

const castTimeAtVideoTime = (videoTime: number): number => {
  if (videoTime <= CAST_TO_VIDEO[0][1]) return CAST_TO_VIDEO[0][0];
  for (let index = 1; index < CAST_TO_VIDEO.length; index++) {
    const [castEnd, videoEnd] = CAST_TO_VIDEO[index];
    const [castStart, videoStart] = CAST_TO_VIDEO[index - 1];
    if (videoTime <= videoEnd) {
      const progress = (videoTime - videoStart) / (videoEnd - videoStart);
      return castStart + progress * (castEnd - castStart);
    }
  }
  return TRAINING_STARTED_CAST_SECONDS;
};

const formatElapsed = (seconds: number): string => {
  const tenths = Math.max(0, Math.round(seconds * 10));
  const minutes = Math.floor(tenths / 600);
  const remainder = tenths - minutes * 600;
  return `${String(minutes).padStart(2, "0")}:${String(Math.floor(remainder / 10)).padStart(2, "0")}.${remainder % 10}`;
};

const Terminal: React.FC<{
  accent: string;
  label: string;
  src: string;
  trimBefore: number;
  freezeAt?: number;
  freezeSrc?: string;
}> = ({accent, label, src, trimBefore, freezeAt, freezeSrc}) => {
  const frame = useCurrentFrame();
  const frozen = freezeAt !== undefined && freezeSrc !== undefined && frame >= freezeAt;
  const mediaStyle: React.CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    objectPosition: "left center",
    scale: 1.24,
    transformOrigin: "left center",
    filter: "grayscale(1) contrast(1.08)",
  };
  return <section style={{flex: 1, minWidth: 0}}>
    <div style={{display: "flex", alignItems: "baseline", margin: "0 4px 10px"}}>
      <div style={{color: accent, fontSize: 24, fontWeight: 760}}>{label}</div>
    </div>
    <div
      style={{
        height: 706,
        background: PANEL,
        border: `1px solid ${RULE}`,
        borderRadius: 14,
        overflow: "hidden",
      }}
    >
      {frozen ? (
        <Img src={staticFile(freezeSrc)} style={mediaStyle} />
      ) : (
        <Video trimBefore={trimBefore} src={staticFile(src)} muted style={mediaStyle} />
      )}
    </div>
  </section>;
};

const milestones = [
  {label: "REQUESTS STOP", elapsed: 0},
  {label: "QUIET CONFIRMED", elapsed: IDLE_DETECTED_CAST_SECONDS - LAST_REQUEST_CAST_SECONDS},
  {label: "vLLM ASLEEP", elapsed: VLLM_ASLEEP_CAST_SECONDS - LAST_REQUEST_CAST_SECONDS},
  {label: "TUNING STARTS", elapsed: TRAINING_STARTED_CAST_SECONDS - LAST_REQUEST_CAST_SECONDS},
] as const;

export const PtyIdleScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const sourceVideoSeconds = SPEEDLM_TRIM_FRAMES / fps + frame / fps;
  const castSeconds = castTimeAtVideoTime(sourceVideoSeconds);
  const elapsed = Math.min(
    TRAINING_STARTED_CAST_SECONDS - LAST_REQUEST_CAST_SECONDS,
    Math.max(0, castSeconds - LAST_REQUEST_CAST_SECONDS),
  );
  const activeMilestone = milestones.reduce(
    (latest, milestone, index) => (elapsed >= milestone.elapsed ? index : latest),
    0,
  );

  return (
    <AbsoluteFill style={{background: BG, padding: "34px 58px", fontFamily: MONO}}>
      <div style={{display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 15}}>
        <div style={{color: TEXT, fontSize: 43, fontWeight: 740}}>
          Traffic stops. Only SpeedLM starts idle tuning.
        </div>
        <div style={{color: DIM, fontSize: 16}}>
          authentic PTYs · demo threshold 5 s · software default 5 min · waits time-lapsed
        </div>
      </div>

      <div
        style={{
          height: 104,
          display: "flex",
          alignItems: "stretch",
          gap: 26,
          marginBottom: 17,
          padding: "13px 18px",
          border: `1px solid ${RULE}`,
          borderRadius: 14,
          background: PANEL,
        }}
      >
        <div style={{width: 300, borderRight: `1px solid ${RULE}`}}>
          <div style={{color: DIM, fontSize: 14, letterSpacing: 1.2}}>ACTUAL ELAPSED</div>
          <div style={{color: TEXT, fontSize: 46, fontWeight: 800, lineHeight: 1.18}}>
            {formatElapsed(elapsed)}
          </div>
        </div>
        <div style={{display: "flex", flex: 1, alignItems: "center", gap: 10}}>
          {milestones.map((milestone, index) => {
            const reached = index <= activeMilestone;
            const active = index === activeMilestone;
            return (
              <React.Fragment key={milestone.label}>
                {index > 0 ? <div style={{height: 2, flex: 1, background: reached ? SERIES_GREEN : RULE}} /> : null}
                <div style={{minWidth: index === 3 ? 168 : 145}}>
                  <div style={{color: active ? SERIES_GREEN : reached ? TEXT : DIM, fontSize: 15, fontWeight: 760}}>
                    {milestone.label}
                  </div>
                  <div style={{color: active ? TEXT : DIM, fontSize: 18, marginTop: 5}}>
                    +{formatElapsed(milestone.elapsed)}
                  </div>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <div style={{display: "flex", gap: 22}}>
        <Terminal
          accent={SERIES_BLUE}
          label="Normal vLLM · ready · no tuner starts"
          src="pty-normal-server.mp4"
          trimBefore={282}
          freezeAt={60}
          freezeSrc="pty-normal-idle.png"
        />
        <Terminal
          accent={SERIES_GREEN}
          label="SpeedLM · waits, then tunes automatically"
          src="pty-speedlm-server.mp4"
          trimBefore={SPEEDLM_TRIM_FRAMES}
        />
      </div>
    </AbsoluteFill>
  );
};
