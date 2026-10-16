import React from "react";
import {Video} from "@remotion/media";
import {
  AbsoluteFill,
  Easing,
  Img,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import {BG, DIM, MONO, PANEL, RULE, SERIES_BLUE, SERIES_GREEN, TEXT} from "../theme";

const clamp = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
};

const TerminalShell: React.FC<{
  accent: string;
  children: React.ReactNode;
  label: string;
  height: number;
}> = ({accent, children, label, height}) => (
  <section style={{flex: 1, minWidth: 0}}>
    <div style={{color: accent, fontSize: 22, fontWeight: 760, margin: "0 0 8px 4px"}}>{label}</div>
    <div
      style={{
        position: "relative",
        height,
        background: PANEL,
        border: `1px solid ${RULE}`,
        borderRadius: 14,
        overflow: "hidden",
      }}
    >
      {children}
    </div>
  </section>
);

const terminalGrade: React.CSSProperties = {
  filter: "grayscale(1) contrast(1.08)",
};

export const PtyServeScene: React.FC = () => {
  const frame = useCurrentFrame();
  const commandOpacity = interpolate(frame, [52, 78], [1, 0], {
    ...clamp,
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const commandScale = interpolate(frame, [0, 52, 78], [1, 1, 0.84], {
    ...clamp,
    easing: Easing.bezier(0.16, 1, 0.3, 1),
    output: "perceptual-scale",
  });
  const serverOpacity = interpolate(frame, [58, 82], [0, 1], {...clamp});
  const serverScale = interpolate(frame, [58, 88], [1.65, 1.15], {
    ...clamp,
    easing: Easing.bezier(0.16, 1, 0.3, 1),
    output: "perceptual-scale",
  });
  const serverHeight = interpolate(frame, [225, 255], [790, 342], {
    ...clamp,
    easing: Easing.bezier(0.16, 1, 0.3, 1),
  });
  const curlOpacity = interpolate(frame, [248, 268], [0, 1], {...clamp});

  const commandStill = (src: string) => (
    <Img
      src={staticFile(src)}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        objectFit: "contain",
        opacity: commandOpacity,
        scale: commandScale,
        ...terminalGrade,
      }}
    />
  );

  const serverVideo = (src: string) => (
    <Video
      src={staticFile(src)}
      trimBefore={6}
      muted
      style={{
        width: "100%",
        height: "100%",
        objectFit: "contain",
        opacity: serverOpacity,
        scale: serverScale,
        transformOrigin: "left top",
        ...terminalGrade,
      }}
    />
  );

  const curlVideo = (src: string) => (
    <Video
      src={staticFile(src)}
      muted
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        objectPosition: "left center",
        ...terminalGrade,
      }}
    />
  );

  return (
    <AbsoluteFill style={{background: BG, padding: "34px 58px 40px", fontFamily: MONO}}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14}}>
        <div style={{color: TEXT, fontSize: 43, fontWeight: 740}}>
          {frame < 82 ? "Keep your vLLM workflow. Add one flag." : frame < 248 ? "Both start as normal vLLM servers" : "Same client API. One matching recorded reply."}
        </div>
        <div style={{color: DIM, fontSize: 16}}>two H100s · literal PTY capture</div>
      </div>

      <div style={{display: "flex", gap: 22, height: serverHeight + 36}}>
        <TerminalShell accent={SERIES_BLUE} label="Your current vLLM" height={serverHeight}>
          {commandStill("pty-normal-command.png")}
          <Sequence from={58}>{serverVideo("pty-normal-server.mp4")}</Sequence>
        </TerminalShell>
        <TerminalShell accent={SERIES_GREEN} label="SpeedLM · familiar serving command" height={serverHeight}>
          {commandStill("pty-speedlm-command.png")}
          <Sequence from={58}>{serverVideo("pty-speedlm-server.mp4")}</Sequence>
        </TerminalShell>
      </div>

      <Sequence from={248} layout="none">
        <div style={{display: "flex", gap: 22, height: 390, marginTop: 20, opacity: curlOpacity}}>
          <TerminalShell accent={SERIES_BLUE} label="Normal vLLM · recorded reply" height={350}>
            {curlVideo("pty-normal-client.mp4")}
          </TerminalShell>
          <TerminalShell accent={SERIES_GREEN} label="SpeedLM · recorded reply" height={350}>
            {curlVideo("pty-speedlm-client.mp4")}
          </TerminalShell>
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
