"""Metrics harness for pyDeltaRCM demo scenarios.

The harness runs scenario YAML files through ``pyDeltaRCM.DeltaModel``,
driving the update loop itself so that per-step wall time and workload are
observable, and records one JSON metrics record per run plus optional
aggregate CSV/Markdown summaries.

It deliberately lives outside the ``pyDeltaRCM`` package: pytest collects
doctests from package modules and coverage scopes the package, and demo
tooling must stay out of both.

Usage::

    python -m harness.run_scenario scenarios/00_smoke.yml --out runs/
    python -m harness.run_scenario scenarios/*.yml --out runs/ --summary
"""
