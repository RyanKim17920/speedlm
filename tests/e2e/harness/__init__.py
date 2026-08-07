"""Systematic benchmarking harness: workloads, results index, comparison, preflight.

This package is deliberately import-light and GPU-free.  Everything here runs in
the project venv on a login node so that the parts of a benchmark that can be
wrong *before* an allocation is spent -- a misconfigured flavor, an
unrepresentative workload, a comparison that cannot resolve its own delta --
fail on a laptop rather than four minutes into an H100 job.
"""
