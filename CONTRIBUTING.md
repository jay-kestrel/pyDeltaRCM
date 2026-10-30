# Contributing to pyDeltaRCM

Thanks for your interest in contributing!

This repository is the Kestrel fork of [DeltaRCM/pyDeltaRCM](https://github.com/DeltaRCM/pyDeltaRCM). The [fork orientation page](docs/source/meta/about_this_fork.rst) — also in the rendered docs as *About this fork* — describes what is shared with upstream and what is fork-specific.

## Where the full contributing process lives

The canonical contribution process — including the [code of conduct](docs/source/meta/conduct.rst) — is maintained in the Sphinx documentation: [`docs/source/meta/contributing.rst`](docs/source/meta/contributing.rst). This root file does not duplicate that process; it adds fork-specific guidance only. (The issue-tracker links in the docs contributing page point at the upstream `DeltaRCM/pyDeltaRCM` issue tracker; file fork-specific issues on this repository.)

## Development quickstart

Install in development mode and run the test suite:

```console
$ git clone https://github.com/jay-kestrel/pyDeltaRCM.git
$ cd pyDeltaRCM
$ pip install -e .
$ pip install -r requirements-test.txt
$ python -m pytest --mpl --mpl-baseline-path=tests/imgs_baseline
```

To build the documentation locally (the LaTeX packages listed in the developer guide are needed to render the example figures):

```console
$ pip install -r requirements.txt -r requirements-docs.txt
$ (cd docs && make docs)
```

To measure model performance with the fork's benchmark harness (a smoke run takes seconds):

```console
$ python benchmarks/benchmark_timestep.py --smoke
```

See [`docs/source/guides/benchmarking.rst`](docs/source/guides/benchmarking.rst) for the full guide, including the scenario design and how CI runs it.

## Tests and docstring doctests

The pytest configuration includes `--doctest-modules`, so docstring examples run as part of the test matrix on every platform and Python version. If you edit docstrings or their examples, run pytest — not only the docs build — before pushing. See the developer guide's CI section for details.

## Commit and pull request conventions

* Open pull requests against the `develop` branch.
* Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat:`, `fix:`, `docs:`, `bench:`, ...), matching the repository's existing history.
* Pull requests are squash-merged; make sure CI is green before merging.

## Fork vs upstream

* Changes that fix or improve the upstream model itself are best proposed upstream, at `DeltaRCM/pyDeltaRCM`, so they benefit both projects.
* Fork-specific tooling and documentation — the benchmark harness and the docs listed on the orientation page — live here.

Performance-oriented pull requests on this fork must report before/after numbers from the benchmark harness against a recorded baseline; there is no speedup claim without one (see [`benchmarks/README.md`](benchmarks/README.md)).
