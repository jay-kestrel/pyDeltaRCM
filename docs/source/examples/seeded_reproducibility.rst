Seeded reproducibility
======================

Simulations run with a fixed random ``seed`` are *bit-reproducible*:
fresh runs of the same configuration produce model states that are
identical, byte for byte. This page demonstrates that guarantee
directly, by hashing the raw bytes of the model state produced by
several fresh runs of one seeded configuration.

The guarantee is also pinned by the integration test suite
(``tests/integration/test_consistent_outputs.py``), which asserts the
exact values of the bed elevation for a seeded simulation. Unlike that
test, the example below is runnable by anyone, and shows the
reproducibility of the whole model state: the bed elevation (``eta``),
the flow depth (``depth``), and the discharge components (``qx`` and
``qy``).

The script launches four **fresh subprocesses** of the same
configuration file, ``seeded_reproducibility.yaml`` (``seed: 14``):
three identical runs (A, B, and C), and a *control run* that overrides
the seed with ``15``. After every model update, the script computes a
SHA-256 digest over the raw bytes of the four state arrays. Identical
digests at every interval mean the runs produced bit-identical states.

.. literalinclude:: seeded_reproducibility.yaml
   :language: yaml

.. literalinclude:: seeded_reproducibility.py
   :language: python

.. program-output:: python examples/seeded_reproducibility.py

The three seeded runs (A, B, and C) produce identical digests at every
save interval, while the control run diverges from the very first
interval: once the seed changes, the entire trajectory of the delta
evolves differently. The configuration used here is intentionally
small (a 50 :math:`\times` 50 cell domain, 10 updates, and reduced
parcel counts) so that the documentation build stays fast; widening
the domain or the parcel counts changes the digest values, but never
the reproducibility guarantee.

.. note::

   Digests depend on the environment (Python, numpy, and numba
   versions, and platform floating-point behavior). They are stable
   across runs of the *same* environment; across environments the
   digest values may differ, in which case the three runs remain
   identical to one another on that platform. The example documents
   this honestly: if the three seeded runs ever disagree, the script
   fails loudly rather than passing silently.

.. hint::

   Use ``seed: <integer>`` in your YAML configuration to make any
   simulation reproducible. Omitting ``seed`` selects a random seed,
   so two runs of the same unseeded configuration will differ.
