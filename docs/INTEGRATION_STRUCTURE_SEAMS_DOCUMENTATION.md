# Integration tests: structural seams

The three files under `tests/integration/` whose subject is the library's own
structure rather than a value moving through it — whether an interface is
genuinely swappable where a model uses it, whether a computation gives the same
answer split across machines as it does serially, and whether every header can
be included in one translation unit at all. These are the ones to read before
changing an interface, adding a header, or distributing a run.

The directory's scope rule, the two rules specific to it, how to run them and
how they are built are in `docs/INTEGRATION_TESTS_DOCUMENTATION.md`, which also
indexes every file here. The four files whose subject is a value crossing
between modules are in `docs/INTEGRATION_DATA_SEAMS_DOCUMENTATION.md`.

## `distributed_simulation.c` — does a Monte Carlo across machines give the serial answer

`cluster/cluster.h` is well tested at the protocol level: real sockets, a
killed worker, discovery, framing, a strided input. Every task in that suite is
a pure arithmetic loop. The one workload anybody would distribute is a
simulation, and that is the case where being wrong leaves no trace — if the
ranges handed to different machines all seed the same generator, every machine
draws the same numbers and the quantile comes out of a sample a fraction of the
size it claims. The engine's half of the contract is `chunk->lo`;
`test_cluster.c` proves `lo` arrives, but none of its tasks draws anything, so
it cannot prove that using it gives back the serial answer.

So: one task, a Gaussian random walk drawn from `rng_new(seed, global index)`
and its ADF statistic, run four ways.

- **serial** — an ordinary loop in this process, the reference.
- **distributed** — `cluster_map` over two workers on loopback sockets, chunk
  size 3 so the split is not trivial. Required to match the serial arm to
  `1e-15` relative, replication by replication, and to agree on the simulated
  5 per cent critical value. Also required to have computed some replications
  off this process, checked through the pid the task records, so a run where
  every range quietly stayed local cannot pass.
- **wrong on purpose** — the same task seeded with a constant stream, required
  to produce one value repeated. This is the negative control: without it the
  distributed check would pass just as happily against a library where every
  draw was identical.
- **after a loss** — a worker killed before the job starts, so its ranges are
  reclaimed and recomputed. Required to produce the same numbers, which is the
  half of reclaiming that the protocol test does not cover: it checks that
  every index is accounted for, not that the recomputed range carries the same
  values.

`STRESS=1` repeats the comparison at 2000 replications and requires exact
equality on every one.

**Result: no defect found.** The distributed and serial arms agree bit for bit.

One hazard flagged during the audit turned out not to exist. The task registry
is a file-static array and a task is identified across machines by its index in
it, so two binaries registering the same number of tasks in a different order
would appear to pass the handshake's task-count check. They do not:
`_cluster_hello_matches` also compares a fingerprint of the whole executable
(`_cluster_self_fingerprint`, an FNV hash of `/proc/self/exe`), and two binaries
with different registration order are different executables. The hole is only
where that fingerprint cannot be read, where it is 0 on both sides and the check
degrades to protocol and ABI — which the header already documents. No test was
written for it.

## `optimizer_swap.c` — is the Optimizer interface swappable where a model uses it

`mlp_fit` takes an `OptimizerInit` and builds one `Optimizer` per trainable
tensor, which is why `solver/optimizer.h` exists as a separate interface. Every
call to `mlp_fit` in the repository passes `adam_optimizer_init`.
`tests/correctness/test_optimizer.c` does build a second implementation, a
stateless SGD, but only to step a bare `Mat` — it never reaches a model.

The optimizer here is SGD with momentum, deliberately not a copy of the
stateless one next door: momentum keeps a velocity buffer of the parameter's
own shape, allocated in `init` from the `(r, c)` `mlp_fit` passes, carried
across every step, and released exactly once in `free`. Those are the three
things a stateless optimizer cannot check.

What is asserted: XOR is learned to the same loss threshold the Adam test uses;
one instance per trainable tensor and one free per instance, counted rather than
assumed, with a counter for any instance built at the wrong shape and an assert
against a step on a freed one; the step count is exactly tensors x epochs x
samples; momentum at 0.9 and at 0 do not land in the same place, which is what
proves the state persists rather than being rebuilt; and the fitted model
round-trips through `mlp_save`/`mlp_load` unchanged.

**Result: no defect found.** The interface holds for an optimizer with
per-tensor state.

## `header_composition.c` and `header_composition_reverse.c` — can every header be included together

This project is header-only and C has one flat namespace. Nothing anywhere
includes more than four of these headers at once, and the four that do are from
the same family. So two headers that both define a function named `fit`, an
object-like macro redefined with different text, or a header that only compiles
because whatever included it first pulled in `<string.h>`, are all invisible
today. `README.md`'s "Implementing a new model" policy anticipates the first of
those and nothing enforced it.

Two translation units on purpose. One includes every header in declaration
order; the other includes them in the opposite order, and the two are linked
into one binary. Two units rather than two orders in one file, because a
duplicate external symbol is a link-time failure and one unit cannot produce
one. The reverse order matters separately: a header that compiles only when
something else came first passes in one order and fails in the other.

Compiling and linking is most of the assertion. What runs is one call per
module, chosen to be the cheapest thing each header offers, so the linker has
to resolve a symbol from every one of them rather than discarding the lot as
unused. Built and run at both precisions.

**Result: no collision found.** All 35 headers compose, in both orders, at
both precisions.
