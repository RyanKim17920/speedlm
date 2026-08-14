import React from "react";
import { Composition } from "remotion";
import { SpeedLMDemo } from "./SpeedLMDemo";

export const FPS = 30;

/**
 * Length of the recorded terminal capture (public/terminal.mp4).
 * demo-fast/session_fast.mp4 is 60.2s -> 1806 frames at 30fps (the renderer
 * reports it, and it agrees with demo-fast/timing.json's total_frames).
 * Swap in a different capture and update this to match its real frame count.
 */
export const DURATION_IN_FRAMES = 1806;

export const RemotionRoot: React.FC = () => (
  <Composition
    id="SpeedLMDemo"
    component={SpeedLMDemo}
    durationInFrames={DURATION_IN_FRAMES}
    fps={FPS}
    width={1920}
    height={1080}
  />
);
