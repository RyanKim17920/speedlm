"""GPU-free full-cycle simulation of the idle-tuning lifecycle.

Everything under this package drives *production* code -- the real
:class:`~speedlm.tuner.orchestrator.TunerOrchestrator`, the real
:class:`~speedlm.tuner.state.TunerStateMachine`, the real
:class:`~speedlm.tuner.artifacts.ArtifactRegistry`, the real
:class:`~speedlm.gate.runner.BenchmarkGateRunner` and the real
:func:`~speedlm.gate.decide.decide_promotion` -- against a simulated vLLM
engine that speaks real HTTP.  Nothing here imports torch, and nothing here
needs a GPU.

The seams it plugs into are the three protocols the whole GPU surface already
sits behind: :class:`~speedlm.training.base.SpeculatorBackend`,
:class:`~speedlm.tuner.orchestrator.RuntimeController` and
:class:`~speedlm.tuner.orchestrator.BenchmarkGate`.
"""

from __future__ import annotations
