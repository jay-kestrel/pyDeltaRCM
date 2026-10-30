# Changelog

All notable changes made on this repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This changelog was seeded when the fork's documentation was written (October 2026), at the version current in the tree at that time (2.3.0, per `pyDeltaRCM/_version.py`). Development history before the seeding belongs to the upstream project, [DeltaRCM/pyDeltaRCM](https://github.com/DeltaRCM/pyDeltaRCM), whose releases and tags record it; this fork tracks upstream's `develop` branch (see the [fork orientation page](docs/source/meta/about_this_fork.rst)).

## [Unreleased]

### Added

- Two-regime per-timestep benchmark harness (`benchmarks/benchmark_timestep.py`, with `benchmarks/README.md`), measuring per-update wall time in water-dominated and sediment-dominated regimes, with a CI smoke preset run by the `benchmark.yml` workflow. (Kestrel PR #1)
- Comprehensive documentation set: fork orientation page (`docs/source/meta/about_this_fork.rst`), root contributing guide (`CONTRIBUTING.md`), benchmarking guide (`docs/source/guides/benchmarking.rst`), preprocessor CLI reference (`docs/source/reference/preprocessor/cli.rst`), technical specification (`docs/source/info/technical_specification.rst`), developer-guide CI and release-flow sections, and this changelog.

## [2.3.0] — version at changelog seeding

The tree's version at seeding was 2.3.0 (`pyDeltaRCM/_version.py`). This entry marks the pre-existing upstream version; no fork changes are recorded below it.
