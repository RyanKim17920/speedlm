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

/**
 * Exact 59-second cut for forms with a hard one-minute ceiling.
 *
 * The compact cut keeps the strongest real evidence: captured replay, literal
 * launch and curl PTYs, the authentic idle lifecycle, retained training data,
 * the retained promotion decision, and the documented install command. It
 * intentionally omits the corpus tour rather than flashing unreadable panes.
 */
export const SpeedLMCampaign60: React.FC = () => (
  <Series>
    <Series.Sequence name="Hosted versus local versus SpeedLM" durationInFrames={330} premountFor={30}>
      <ProblemSolutionScene />
    </Series.Sequence>
    <Series.Sequence name="Archived accepted-length result" durationInFrames={210} premountFor={30}>
      <HookScene />
    </Series.Sequence>
    <Series.Sequence name="Literal commands and server PTYs" durationInFrames={270} premountFor={45}>
      <LaunchScene />
    </Series.Sequence>
    <Series.Sequence name="Literal matching curl responses" durationInFrames={270} premountFor={30}>
      <CurlProofScene />
    </Series.Sequence>
    <Series.Sequence name="Authentic idle lifecycle" durationInFrames={300} premountFor={45}>
      <IdleSubmissionScene />
    </Series.Sequence>
    <Series.Sequence name="Retained training metrics" durationInFrames={120} premountFor={30}>
      <RetainedTrainingScene revealEndFrame={75} />
    </Series.Sequence>
    <Series.Sequence name="Retained promotion gate" durationInFrames={150} premountFor={30}>
      <RetainedGateScene promotedFrame={55} revealEndFrame={60} revealStartFrame={15} />
    </Series.Sequence>
    <Series.Sequence name="Quick start" durationInFrames={120} premountFor={30}>
      <QuickStartScene />
    </Series.Sequence>
  </Series>
);

/**
 * Exact 1:59 cut for forms with a hard two-minute ceiling.
 *
 * This retains every evidence category from the 2:30 master. The four real
 * traffic excerpts and both verbatim trace examples are sped up structurally,
 * while terminal pixels and measured values remain untouched.
 */
export const SpeedLMCampaign120: React.FC = () => (
  <Series>
    <Series.Sequence name="Hosted versus local versus SpeedLM" durationInFrames={300} premountFor={30}>
      <ProblemSolutionScene />
    </Series.Sequence>
    <Series.Sequence name="Archived accepted-length result" durationInFrames={240} premountFor={30}>
      <HookScene />
    </Series.Sequence>
    <Series.Sequence name="Verifier architecture" durationInFrames={150} premountFor={30}>
      <VerifierScene />
    </Series.Sequence>
    <Series.Sequence name="Literal commands and server PTYs" durationInFrames={420} premountFor={45}>
      <LaunchScene />
    </Series.Sequence>
    <Series.Sequence name="Literal matching curl responses" durationInFrames={360} premountFor={30}>
      <CurlProofScene />
    </Series.Sequence>
    <Series.Sequence name="Thirty-one minute traffic proof" durationInFrames={270} premountFor={30}>
      <TrafficProofScene segmentFrames={68} />
    </Series.Sequence>
    <Series.Sequence name="Verbatim trace examples" durationInFrames={390} premountFor={30}>
      <TraceProofScene sampleFrames={195} />
    </Series.Sequence>
    <Series.Sequence name="Authentic idle lifecycle" durationInFrames={450} premountFor={45}>
      <IdleSubmissionScene />
    </Series.Sequence>
    <Series.Sequence name="Retained training metrics" durationInFrames={360} premountFor={30}>
      <RetainedTrainingScene revealEndFrame={300} />
    </Series.Sequence>
    <Series.Sequence name="Retained promotion gate" durationInFrames={390} premountFor={30}>
      <RetainedGateScene promotedFrame={210} revealEndFrame={240} revealStartFrame={90} />
    </Series.Sequence>
    <Series.Sequence name="Quick start" durationInFrames={240} premountFor={30}>
      <QuickStartScene />
    </Series.Sequence>
  </Series>
);
