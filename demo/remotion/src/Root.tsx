import React from "react";
import { Composition } from "remotion";
import { SpeedLMDemo } from "./SpeedLMDemo";
import { SpeedLMShort } from "./short/SpeedLMShort";
import { SpeedLMCampaign120, SpeedLMCampaign60 } from "./submission/SpeedLMCampaignCuts";
import { SpeedLMSubmission } from "./submission/SpeedLMSubmission";

export const FPS = 30;

/**
 * Length of the recorded terminal capture (public/terminal.mp4).
 * demo-fast/session_fast_v4.mp4 is 44.83s -> 1344 frames at 30fps (the renderer
 * reports it, and it agrees with demo-fast/timing_v4.json's total_frames).
 * Swap in a different capture and update this to match its real frame count.
 */
export const DURATION_IN_FRAMES = 1349;

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="SpeedLMDemo"
      component={SpeedLMDemo}
      durationInFrames={DURATION_IN_FRAMES}
      fps={FPS}
      width={1920}
      height={1080}
    />
    <Composition
      id="SpeedLMShort"
      component={SpeedLMShort}
      durationInFrames={1902}
      fps={FPS}
      width={1920}
      height={1080}
    />
    <Composition
      id="SpeedLMSubmission"
      component={SpeedLMSubmission}
      durationInFrames={4500}
      fps={FPS}
      width={1920}
      height={1080}
    />
    <Composition
      id="SpeedLMCampaign120"
      component={SpeedLMCampaign120}
      durationInFrames={3570}
      fps={FPS}
      width={1920}
      height={1080}
    />
    <Composition
      id="SpeedLMCampaign60"
      component={SpeedLMCampaign60}
      durationInFrames={1770}
      fps={FPS}
      width={1920}
      height={1080}
    />
  </>
);
