Checkpoint/resume continuity
============================

pyDeltaRCM can save its complete state --- grids, clocks, and the state of
its random number generator --- to a checkpoint file, and a new model can
then resume from that checkpoint as if the run had never stopped. This
example demonstrates that guarantee: a run interrupted mid-way and resumed
from its checkpoint in a **fresh interpreter** reproduces the state
trajectory of an uninterrupted run, bit for bit.

How the example works
---------------------

One script executes three model runs from the same seeded configuration:

1. a **reference** run, advanced for eight updates without interruption,
2. an **interrupted** run, advanced for four updates with
   :code:`save_checkpoint: true` and :code:`checkpoint_dt: 50000`
   (checkpoints are saved every two updates at the derived model timestep
   of this configuration, :code:`dt` = 25000 s), and
3. a **resumed** run, started in a fresh interpreter process from the
   interrupted run's checkpoint, then advanced to the reference's total
   update count.

After every update, each run hashes its state --- the raw bytes of the
:code:`eta`, :code:`depth`, :code:`qx` and :code:`qy` arrays --- with
sha256. The script asserts that the hashes of the interrupted/resumed
trajectory match the reference at every compared update, and prints the
comparison table below.

.. literalinclude:: checkpoint_resume_continuity.py
   :language: python

.. program-output:: python examples/checkpoint_resume_continuity.py

The resumed trajectory is bit-identical to the uninterrupted reference at
every compared update. This includes the update immediately following the
resume, which is only possible because the checkpoint restores the random
number generator state alongside the model state: without it, the resumed
run would replay the random sequence from the start of the simulation and
diverge at the first routed parcel.

The specific hash values are environment-dependent: they differ across
platforms, and between the Python reference routing path and the optional
native water extension. What the example demonstrates is their equality
within one environment.

Notes and caveats
-----------------

This example demonstrates the **direct-DeltaModel** resume path (a model
constructed with :code:`resume_checkpoint: true`, following the pattern in
:doc:`resume_from_checkpoint`). Three checkpoint-integrity fixes formerly
lived on the unmerged :code:`fix/checkpoint-resume-integrity` branch and
were documented as caveats in earlier drafts of this example: serial
:code:`Preprocessor` jobs re-seeded the random number generator after a
resume, clobbering the restored state; the checkpoint-cadence counter was
not persisted, so a resumed run re-triggered its first checkpoint
immediately; and the netCDF-output fallback on resume raised a
:class:`TypeError`. **These have since been merged to ``develop``** (PR
`#3 <https://github.com/jay-kestrel/pyDeltaRCM/pull/3>`__, preserving the
restored RNG state in serial resume and saving/restoring the cadence
counter; the rng state is stored as plain typed arrays and legacy pickled
checkpoints are refused in PR
`#11 <https://github.com/jay-kestrel/pyDeltaRCM/pull/11>`__).

Scope of the claim:

- The example exercises a single stop/resume cycle on the direct
  :code:`DeltaModel` path. It makes no claim about chained multi-cycle
  resumes, and it does not itself exercise :code:`Preprocessor`-driven
  resume (the serial-Preprocessor path is covered by its own regression
  test from PR #3).
- Checkpoint files written by pyDeltaRCM <= 2.3 stored the random state as
  a pickled object array; loading such a file executes the code embedded
  in it, so current versions refuse these files outright. Regenerate old
  checkpoints by re-running the model with this pyDeltaRCM version.
- :code:`checkpoint_dt` is expressed in seconds of *model time*, and the
  first update of any run always writes a checkpoint (the checkpoint
  counter starts at infinity by design, forcing a save at :code:`t == 0`).
- Each of the three processes needs its own output location, so the
  driver writes per-run copies of the YAML with :code:`out_dir` pointing
  at temporary directories. Passing :code:`out_dir` as a keyword argument
  also works on current ``develop`` (fixed in PR
  `#25 <https://github.com/jay-kestrel/pyDeltaRCM/pull/25>`__); older
  releases read :code:`out_dir` from the configuration file before
  merging keyword arguments and ignore the keyword.

The configuration is tuned for documentation-build speed (small domain,
few parcels, eight updates) rather than geomorphic realism --- treat it as
a template, and widen the runs as needed.
