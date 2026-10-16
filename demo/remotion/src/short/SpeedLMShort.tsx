import React from "react";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { OutroScene } from "./OutroScene";
import { PtyIdleScene } from "./PtyIdleScene";
import { PtyServeScene } from "./PtyServeScene";
import { RaceScene } from "./RaceScene";
import { ResultScene } from "./ResultScene";
import { TrafficStreamScene } from "./TrafficStreamScene";
import { TrainingScene } from "./TrainingScene";

export const SpeedLMShort: React.FC = () => (
  <TransitionSeries name="SpeedLM short demo">
    <TransitionSeries.Sequence name="Trimmed stock vs SpeedLM output race" durationInFrames={240}>
      <RaceScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Captured replay proof" durationInFrames={120}>
      <ResultScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Workflow unchanged" durationInFrames={90}>
      <OutroScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Literal server and curl PTY recordings" durationInFrames={450}>
      <PtyServeScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Simplified captured input-output stream" durationInFrames={300}>
      <TrafficStreamScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Literal idle and training server logs" durationInFrames={360}>
      <PtyIdleScene />
    </TransitionSeries.Sequence>
    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 8 })}
    />
    <TransitionSeries.Sequence name="Idle training metrics" durationInFrames={390}>
      <TrainingScene />
    </TransitionSeries.Sequence>
  </TransitionSeries>
);
