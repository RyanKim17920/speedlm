import React from "react";
import {Series} from "remotion";
import {CurlProofScene} from "./CurlProofScene";
import {HookScene} from "./HookScene";
import {IdleSubmissionScene} from "./IdleSubmissionScene";
import {LaunchScene} from "./LaunchScene";
import {ProblemSolutionScene} from "./ProblemSolutionScene";
import {QuickStartScene} from "./QuickStartScene";
import {RetainedGateScene} from "./RetainedGateScene";
import {RetainedTrainingScene} from "./RetainedTrainingScene";
import {TraceProofScene} from "./TraceProofScene";
import {TrafficProofScene} from "./TrafficProofScene";
import {VerifierScene} from "./VerifierScene";

export const SpeedLMSubmission: React.FC = () => (
  <Series>
    <Series.Sequence name="Hosted versus local versus SpeedLM" durationInFrames={330} premountFor={30}>
      <ProblemSolutionScene />
    </Series.Sequence>
    <Series.Sequence name="Authentic replay and archived accepted-length result" durationInFrames={240} premountFor={30}>
      <HookScene />
    </Series.Sequence>
    <Series.Sequence name="Why the verifier stays authoritative" durationInFrames={180} premountFor={30}>
      <VerifierScene />
    </Series.Sequence>
    <Series.Sequence name="Literal launch commands and server PTYs" durationInFrames={480} premountFor={45}>
      <LaunchScene />
    </Series.Sequence>
    <Series.Sequence name="Literal curl requests and matching responses" durationInFrames={450} premountFor={30}>
      <CurlProofScene />
    </Series.Sequence>
    <Series.Sequence name="Thirty-one minute live traffic proof" durationInFrames={330} premountFor={30}>
      <TrafficProofScene segmentFrames={83} />
    </Series.Sequence>
    <Series.Sequence name="Verbatim retained input and predicted output" durationInFrames={600} premountFor={30}>
      <TraceProofScene sampleFrames={300} />
    </Series.Sequence>
    <Series.Sequence name="Literal idle lifecycle and exact elapsed clock" durationInFrames={540} premountFor={45}>
      <IdleSubmissionScene />
    </Series.Sequence>
    <Series.Sequence name="Retained full training run" durationInFrames={540} premountFor={30}>
      <RetainedTrainingScene />
    </Series.Sequence>
    <Series.Sequence name="Retained session-disjoint promotion gate" durationInFrames={510} premountFor={30}>
      <RetainedGateScene />
    </Series.Sequence>
    <Series.Sequence name="Quick start and OpenAI-compatible serving API" durationInFrames={300} premountFor={30}>
      <QuickStartScene />
    </Series.Sequence>
  </Series>
);
