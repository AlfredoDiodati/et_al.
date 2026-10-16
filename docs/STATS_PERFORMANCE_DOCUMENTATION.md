# stats.h - selection performance

How `stats_quickselect`, the selection behind `stats_median`, `stats_quantile`, `stats_quantile_inplace` and `stats_medae`, performs on repeated values, on distinct values in random order and on sorted input; which versions of it were measured; and why the shipped one was chosen. Extracted from `docs/STATS_DOCUMENTATION.md` on the size rule in README's "Documentation structure": it is read when changing the selection, not when calling it.

## The versions

Every version shares everything but the selection loop: the median-of-three pivot (`stats_median_of_three`), the copy loop with its non-finite guard, and the scan of the left part for an even count.

| name | what a pass does | status |
|---|---|---|
| `previous` | two-way partition (Lomuto, `stats_partition`): values below the pivot to the left, everything else to the right | shipped until this change |
| `three_way` | three-way partition (Dijkstra): below, equal to and above the pivot; stops when the middle index falls in the equal group | measured, not shipped |
| `repeat` | `previous`, plus the equal-pivot step of pdqsort: after moving right past a pivot nothing remaining is below it, so when the next pivot equals it one pass (`stats_gather_equal`) gathers the values equal to it at the front, and the selection stops if the middle index is among them | **shipped** |
| `repeat_bl` | `repeat` with both passes branch-free: every element is written and the boundary moves by the 0 or 1 the comparison returns | measured, not shipped |

Why `previous` collapses: a run of values equal to the pivot stays on one side of a two-way partition. When the middle index falls inside that run, each later pass removes one element of it, so the run is scanned once per element it holds and the cost grows with the square of the run's length. `three_way`, `repeat` and `repeat_bl` all gather the run, so it costs one pass however long it is.

Why `repeat` is shipped: it removes the collapse, and its cost on input already sorted was judged acceptable. Run 7, the run on the shipped `stats_median`, has the whole trade in one run. With 1 to 101 distinct values it is 1.04 to 153,000 times faster than `previous` (the median of 2,000,000 equal values went from 2,024 s to 13.2 ms); with 1,001 distinct values and with distinct values in random order it is within noise of `previous`; on input already sorted it is slower, 7% to 18% descending at every length and 1% to 11% ascending at four of six lengths (1,000,000 descending: 4.633 ms against 5.615).

The sorted-input loss appeared only after the deployment. In runs 4 to 6 `repeat` was a copy inside the benchmark, called through a function pointer, and no cell had it below `previous` in both the best and the median round; `stats_median` calls the selection directly. Why the two builds differ is not known, and finding out is part of item 23 of `docs/PERFORMANCE_BACKLOG.md`.

The two alternatives, from run 6: `three_way` loses up to 36% on distinct values in random order, because it swaps every element above the pivot as well as every one below it. `repeat_bl` is about twice as fast as `previous` on distinct values in random order, but 7% to 29% slower on input already sorted ascending. My reading of why, not profiled: the shipped loop's branch goes either way about half of the time on random input and is mispredicted, while on sorted input it always goes the same way and costs nothing, so writing every element unconditionally only pays on random input. Getting `repeat_bl`'s gain without its loss is the rest of item 23.

## How every measurement was taken

Same for every run below unless the run says otherwise.

- Machine: Intel Core i5-7400 @ 3.00 GHz, 4 physical cores, 4 hardware threads, all 4 used; L2 256 KiB per core, L3 6 MiB; 23.4 GiB memory; Linux 6.8.0-147-generic x86_64; gcc 13.3.0; float32 (the default build, `-O3 -march=native -ffast-math -fopenmp`). Each report writes these itself.
- `previous` is `stats.h` as of commit `208a84a`.
- Samples, generated at run time: `n` values from `levels` distinct integers `0, ..., levels - 1` in equal blocks (element `i` of the sorted sample is `floor(i * levels / n)`), put in random order by `rng_permutation` from `rng_new(23, n)`. "distinct" means `levels = n`. The level counts are odd so the middle of the sorted sample falls at the centre of a run of equal values; run 1 shows what an even count does instead. The sorted samples of run 6 are `0, ..., n - 1` ascending or descending.
- Every returned median is checked against the one the construction implies before timing; the candidate runs also check every arm against a full sort on 20,000 short samples (length 1 to 200, `rng_new(31, 0)`, half from 1 to 8 integer values, half standard normal). No run disagreed.
- Measured: ms per call while all 4 threads each run one call on the same read-only sample. The clock is read once per batch; `calls` doubles until a batch takes at least 20 ms.
- Rounds: best of up to 9 timed rounds after one discarded warm-up. A cell stops taking rounds once it has used 30 s (run 2) or 60 s (runs 3 to 6), keeps at least one, and skips the warm-up when one call exceeds 1 s; `rounds` is the count behind each cell. In the candidate runs a round is one batch of every arm, the arm order reversed every other round.
- Speedup: `previous` time over the arm's time, from the best round and from the median round (`best/med`). A speedup whose side of 1 differs between the two is noise, by README's rule.
- Absolute times are not comparable across runs: the same cells measured about twice as fast in runs 4 to 6 as in runs 2 and 3 (the all-equal cell at 2,000,000 took 2,030 s in run 2 and 1,119 s in run 5), for a reason not identified. Ratios within a run are paired and unaffected.

## Run 1: even level counts, `previous` alone (discarded)

`tests/performance/median_repeated_values.c` with levels 1, 2, 10, 100, 1000. Stopped after two sizes: with an even count the middle index falls on the boundary between two runs, where a partition around the upper run's value puts it exactly at the middle index, and the selection runs at distinct-sample speed. That is why the benchmark uses odd counts.

| n | levels | best ms | slowdown over distinct |
|---|---|---|---|
| 1,000 | 1 | 0.467 | 64.0 |
| 1,000 | 2 | 0.006 | 0.8 |
| 1,000 | 10 | 0.007 | 0.9 |
| 1,000 | 100 | 0.006 | 0.8 |
| 1,000 | 1000 | 0.007 | 1.0 |
| 1,000 | distinct | 0.007 | 1.0 |
| 10,000 | 1 | 46.009 | 315.8 |
| 10,000 | 2 | 0.080 | 0.6 |
| 10,000 | 10 | 0.132 | 0.9 |
| 10,000 | 100 | 0.143 | 1.0 |
| 10,000 | 1000 | 0.153 | 1.1 |
| 10,000 | distinct | 0.146 | 1.0 |

## Run 2: `previous` on repeated values

`make bench-median_repeated_values`, 30 s per cell. Slowdown is the cell's best time over the distinct column's at the same `n`.

| n | levels | rounds | best ms | median ms | ns per element | slowdown |
|---|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.467 | 0.542 | 466.87 | 62.7 |
| 1,000 | 3 | 9 | 0.057 | 0.062 | 57.48 | 7.7 |
| 1,000 | 11 | 9 | 0.012 | 0.012 | 11.81 | 1.6 |
| 1,000 | 101 | 9 | 0.008 | 0.011 | 8.40 | 1.1 |
| 1,000 | 1001 | 9 | 0.007 | 0.009 | 7.37 | 1.0 |
| 1,000 | distinct | 9 | 0.007 | 0.008 | 7.45 | 1.0 |
| 10,000 | 1 | 9 | 45.836 | 48.821 | 4583.57 | 314.0 |
| 10,000 | 3 | 9 | 5.205 | 8.320 | 520.55 | 35.7 |
| 10,000 | 11 | 9 | 0.543 | 0.600 | 54.26 | 3.7 |
| 10,000 | 101 | 9 | 0.140 | 0.156 | 13.99 | 1.0 |
| 10,000 | 1001 | 9 | 0.146 | 0.157 | 14.61 | 1.0 |
| 10,000 | distinct | 9 | 0.146 | 0.150 | 14.60 | 1.0 |
| 100,000 | 1 | 6 | 4955.638 | 5045.507 | 49556.38 | 3723.7 |
| 100,000 | 3 | 9 | 532.489 | 576.633 | 5324.89 | 400.1 |
| 100,000 | 11 | 9 | 38.966 | 39.102 | 389.66 | 29.3 |
| 100,000 | 101 | 9 | 2.009 | 2.014 | 20.09 | 1.5 |
| 100,000 | 1001 | 9 | 1.321 | 1.329 | 13.21 | 1.0 |
| 100,000 | distinct | 9 | 1.331 | 1.350 | 13.31 | 1.0 |
| 500,000 | 1 | 1 | 124203.935 | 124203.935 | 248407.87 | 22009.9 |
| 500,000 | 3 | 3 | 13534.984 | 13578.718 | 27069.97 | 2398.5 |
| 500,000 | 11 | 10 | 1011.411 | 1063.870 | 2022.82 | 179.2 |
| 500,000 | 101 | 9 | 18.306 | 18.362 | 36.61 | 3.2 |
| 500,000 | 1001 | 9 | 6.330 | 8.167 | 12.66 | 1.1 |
| 500,000 | distinct | 9 | 5.643 | 5.677 | 11.29 | 1.0 |
| 1,000,000 | 1 | 1 | 507187.124 | 507187.124 | 507187.12 | 43141.9 |
| 1,000,000 | 3 | 1 | 54237.915 | 54237.915 | 54237.92 | 4613.5 |
| 1,000,000 | 11 | 8 | 4038.950 | 4083.310 | 4038.95 | 343.6 |
| 1,000,000 | 101 | 9 | 57.401 | 57.623 | 57.40 | 4.9 |
| 1,000,000 | 1001 | 9 | 13.528 | 13.920 | 13.53 | 1.2 |
| 1,000,000 | distinct | 9 | 11.756 | 11.798 | 11.76 | 1.0 |
| 2,000,000 | 1 | 1 | 2029824.508 | 2029824.508 | 1014912.25 | 66866.4 |
| 2,000,000 | 3 | 1 | 223525.756 | 223525.756 | 111762.88 | 7363.4 |
| 2,000,000 | 11 | 2 | 16127.367 | 16143.177 | 8063.68 | 531.3 |
| 2,000,000 | 101 | 9 | 207.907 | 211.549 | 103.95 | 6.8 |
| 2,000,000 | 1001 | 9 | 29.097 | 29.379 | 14.55 | 1.0 |
| 2,000,000 | distinct | 9 | 30.356 | 30.456 | 15.18 | 1.0 |

The all-equal time quadruples each time `n` doubles from 500,000 on: the cost grows with the square of the run's length. The cell at 500,000 and 11 levels has 10 rounds instead of at most 9, from a counting slip when the warm-up was skipped, fixed after this run; it changes the number of rounds, not how a round is timed.

## Run 3: `previous` against `three_way`

`make bench-median_candidates` with two arms, 60 s per cell. Times are best ms.

| n | levels | rounds | previous | three_way | speedup best/med |
|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.951 | 0.008 | 119.74/117.82 |
| 1,000 | 3 | 9 | 0.116 | 0.009 | 13.39/13.39 |
| 1,000 | 11 | 9 | 0.022 | 0.011 | 2.07/2.24 |
| 1,000 | 101 | 9 | 0.016 | 0.012 | 1.35/1.28 |
| 1,000 | 1001 | 9 | 0.014 | 0.013 | 1.08/1.07 |
| 1,000 | distinct | 9 | 0.014 | 0.013 | 1.04/1.03 |
| 10,000 | 1 | 9 | 75.538 | 0.085 | 893.78/850.52 |
| 10,000 | 3 | 9 | 9.280 | 0.164 | 56.60/50.66 |
| 10,000 | 11 | 9 | 1.030 | 0.255 | 4.04/3.82 |
| 10,000 | 101 | 9 | 0.258 | 0.207 | 1.25/1.25 |
| 10,000 | 1001 | 9 | 0.269 | 0.243 | 1.11/1.14 |
| 10,000 | distinct | 9 | 0.264 | 0.271 | 0.97/0.98 |
| 100,000 | 1 | 9 | 4967.969 | 0.415 | 11971.66/10038.00 |
| 100,000 | 3 | 9 | 549.467 | 1.040 | 528.38/353.86 |
| 100,000 | 11 | 9 | 39.032 | 1.385 | 28.19/29.57 |
| 100,000 | 101 | 9 | 1.957 | 1.572 | 1.24/1.32 |
| 100,000 | 1001 | 9 | 1.258 | 1.593 | 0.79/0.81 |
| 100,000 | distinct | 9 | 1.268 | 1.548 | 0.82/0.68 |
| 500,000 | 1 | 1 | 124101.023 | 2.122 | 58473.64/58473.64 |
| 500,000 | 3 | 5 | 13489.867 | 5.232 | 2578.39/2584.84 |
| 500,000 | 11 | 9 | 1014.350 | 6.698 | 151.45/159.08 |
| 500,000 | 101 | 9 | 18.055 | 6.022 | 3.00/3.00 |
| 500,000 | 1001 | 9 | 6.006 | 7.784 | 0.77/0.77 |
| 500,000 | distinct | 9 | 5.352 | 8.044 | 0.67/0.67 |
| 1,000,000 | 1 | 1 | 507502.818 | 4.331 | 117185.71/117185.71 |
| 1,000,000 | 3 | 2 | 54260.522 | 10.547 | 5144.51/5122.90 |
| 1,000,000 | 11 | 9 | 4022.283 | 13.887 | 289.64/292.56 |
| 1,000,000 | 101 | 9 | 57.062 | 15.602 | 3.66/3.62 |
| 1,000,000 | 1001 | 9 | 12.983 | 11.672 | 1.11/1.11 |
| 1,000,000 | distinct | 9 | 11.199 | 12.962 | 0.86/0.87 |
| 2,000,000 | 1 | 1 | 2036734.590 | 8.730 | 233313.11/233313.11 |
| 2,000,000 | 3 | 1 | 224564.964 | 21.211 | 10587.25/10587.25 |
| 2,000,000 | 11 | 4 | 16091.931 | 21.854 | 736.34/737.45 |
| 2,000,000 | 101 | 9 | 206.754 | 27.970 | 7.39/7.38 |
| 2,000,000 | 1001 | 9 | 28.103 | 29.949 | 0.94/0.95 |
| 2,000,000 | distinct | 9 | 29.306 | 27.469 | 1.07/1.07 |

The `n = 1,000` times of both arms are about twice those of run 2 on the same cells, the reverse of the drift between later runs; the cause is not identified.

## Run 4: all four arms, `n` up to 100,000 (quick check)

`./tests/performance/median_candidates 100000`, four arms, 60 s per cell. Times are best ms; in this run and the next the arm then named `current` was `previous`, and `repeat` was a copy in the benchmark rather than `stats_median`.

| n | levels | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.241 | 0.004 | 0.004 | 0.005 | 60.23/66.27 | 53.69/57.35 | 47.16/44.77 |
| 1,000 | 3 | 9 | 0.033 | 0.004 | 0.005 | 0.005 | 7.98/8.30 | 7.28/7.54 | 6.79/7.59 |
| 1,000 | 11 | 9 | 0.010 | 0.005 | 0.007 | 0.005 | 2.10/1.90 | 1.45/1.48 | 2.02/2.15 |
| 1,000 | 101 | 9 | 0.007 | 0.005 | 0.007 | 0.006 | 1.35/1.43 | 1.08/1.20 | 1.34/1.50 |
| 1,000 | 1001 | 9 | 0.007 | 0.006 | 0.006 | 0.005 | 1.18/1.29 | 1.06/1.15 | 1.26/1.23 |
| 1,000 | distinct | 9 | 0.007 | 0.006 | 0.006 | 0.005 | 1.21/2.32 | 1.08/2.07 | 1.29/2.39 |
| 10,000 | 1 | 9 | 23.049 | 0.039 | 0.044 | 0.047 | 588.44/734.63 | 528.14/656.73 | 491.99/559.30 |
| 10,000 | 3 | 9 | 2.682 | 0.096 | 0.097 | 0.044 | 27.94/31.23 | 27.73/29.67 | 61.45/68.65 |
| 10,000 | 11 | 9 | 0.337 | 0.129 | 0.130 | 0.059 | 2.61/2.92 | 2.59/3.02 | 5.74/6.00 |
| 10,000 | 101 | 9 | 0.124 | 0.114 | 0.118 | 0.052 | 1.09/1.23 | 1.05/1.10 | 2.41/2.80 |
| 10,000 | 1001 | 9 | 0.132 | 0.123 | 0.128 | 0.053 | 1.07/1.16 | 1.03/1.24 | 2.48/2.98 |
| 10,000 | distinct | 9 | 0.132 | 0.138 | 0.128 | 0.058 | 0.95/0.88 | 1.02/1.01 | 2.27/2.27 |
| 100,000 | 1 | 9 | 2512.471 | 0.392 | 0.444 | 0.462 | 6415.55/6246.77 | 5655.70/5065.76 | 5440.13/5441.41 |
| 100,000 | 3 | 9 | 267.151 | 1.034 | 1.002 | 0.478 | 258.42/249.02 | 266.50/262.90 | 559.48/612.41 |
| 100,000 | 11 | 9 | 20.027 | 1.387 | 0.985 | 0.492 | 14.44/13.71 | 20.33/19.68 | 40.71/44.38 |
| 100,000 | 101 | 9 | 1.629 | 1.553 | 1.354 | 0.539 | 1.05/0.91 | 1.20/1.03 | 3.02/3.01 |
| 100,000 | 1001 | 9 | 1.181 | 1.568 | 1.145 | 0.571 | 0.75/0.86 | 1.03/1.14 | 2.07/2.15 |
| 100,000 | distinct | 9 | 1.181 | 1.524 | 1.160 | 0.616 | 0.77/0.71 | 1.02/0.98 | 1.92/1.78 |

## Run 5: all four arms, random order

`make bench-median_candidates`, four arms, 60 s per cell. Times are best ms.

| n | levels | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.241 | 0.004 | 0.004 | 0.005 | 60.22/62.16 | 53.78/53.66 | 51.24/48.54 |
| 1,000 | 3 | 9 | 0.033 | 0.004 | 0.005 | 0.005 | 7.87/7.21 | 7.26/7.17 | 6.74/6.17 |
| 1,000 | 11 | 9 | 0.010 | 0.005 | 0.007 | 0.005 | 2.07/2.13 | 1.44/1.36 | 1.99/2.01 |
| 1,000 | 101 | 9 | 0.007 | 0.006 | 0.007 | 0.006 | 1.33/1.33 | 1.08/0.90 | 1.33/1.26 |
| 1,000 | 1001 | 9 | 0.007 | 0.006 | 0.006 | 0.005 | 1.16/1.16 | 1.04/1.06 | 1.23/1.27 |
| 1,000 | distinct | 9 | 0.007 | 0.006 | 0.006 | 0.005 | 1.18/1.23 | 1.07/1.12 | 1.26/1.32 |
| 10,000 | 1 | 9 | 22.995 | 0.039 | 0.044 | 0.046 | 587.34/586.60 | 526.63/526.01 | 499.45/488.59 |
| 10,000 | 3 | 9 | 2.669 | 0.095 | 0.096 | 0.043 | 27.96/27.76 | 27.70/27.59 | 61.40/61.30 |
| 10,000 | 11 | 9 | 0.337 | 0.129 | 0.129 | 0.059 | 2.60/2.61 | 2.61/2.60 | 5.74/5.74 |
| 10,000 | 101 | 9 | 0.124 | 0.113 | 0.118 | 0.051 | 1.10/1.28 | 1.05/1.22 | 2.41/2.79 |
| 10,000 | 1001 | 9 | 0.131 | 0.124 | 0.128 | 0.053 | 1.05/1.06 | 1.02/1.03 | 2.45/2.48 |
| 10,000 | distinct | 9 | 0.131 | 0.138 | 0.129 | 0.058 | 0.95/0.95 | 1.02/1.02 | 2.27/2.28 |
| 100,000 | 1 | 9 | 2440.928 | 0.392 | 0.436 | 0.461 | 6225.24/6413.60 | 5601.52/5779.75 | 5299.85/5467.41 |
| 100,000 | 3 | 9 | 259.310 | 1.027 | 0.999 | 0.478 | 252.49/285.96 | 259.50/291.73 | 542.60/616.63 |
| 100,000 | 11 | 9 | 20.007 | 1.382 | 0.985 | 0.492 | 14.48/14.37 | 20.31/20.28 | 40.66/40.71 |
| 100,000 | 101 | 9 | 1.629 | 1.550 | 1.353 | 0.539 | 1.05/1.05 | 1.20/1.20 | 3.02/3.03 |
| 100,000 | 1001 | 9 | 1.176 | 1.564 | 1.142 | 0.570 | 0.75/0.76 | 1.03/1.04 | 2.06/2.08 |
| 100,000 | distinct | 9 | 1.181 | 1.515 | 1.159 | 0.616 | 0.78/0.78 | 1.02/1.02 | 1.92/1.17 |
| 500,000 | 1 | 1 | 64528.012 | 2.032 | 2.619 | 2.472 | 31749.01/31749.01 | 24634.77/24634.77 | 26104.15/26104.15 |
| 500,000 | 3 | 9 | 6756.829 | 5.207 | 5.213 | 2.383 | 1297.70/1298.39 | 1296.16/1292.04 | 2835.49/2709.99 |
| 500,000 | 11 | 9 | 510.854 | 6.734 | 5.996 | 2.975 | 75.86/80.10 | 85.20/89.91 | 171.70/181.81 |
| 500,000 | 101 | 9 | 12.049 | 5.950 | 6.210 | 2.830 | 2.03/2.03 | 1.94/1.94 | 4.26/4.25 |
| 500,000 | 1001 | 9 | 5.681 | 7.698 | 5.430 | 2.867 | 0.74/0.74 | 1.05/1.06 | 1.98/2.00 |
| 500,000 | distinct | 9 | 5.093 | 7.954 | 4.929 | 2.798 | 0.64/0.64 | 1.03/1.03 | 1.82/1.83 |
| 1,000,000 | 1 | 1 | 276025.458 | 4.189 | 4.843 | 10.174 | 65891.47/65891.47 | 56994.72/56994.72 | 27130.05/27130.05 |
| 1,000,000 | 3 | 3 | 27553.633 | 10.551 | 10.547 | 5.609 | 2611.44/2598.71 | 2612.36/2614.29 | 4912.47/4907.41 |
| 1,000,000 | 11 | 9 | 2035.891 | 13.950 | 10.642 | 5.322 | 145.94/150.07 | 191.30/196.89 | 382.51/393.38 |
| 1,000,000 | 101 | 9 | 33.839 | 15.505 | 11.069 | 6.519 | 2.18/2.19 | 3.06/3.06 | 5.19/5.19 |
| 1,000,000 | 1001 | 9 | 12.026 | 11.490 | 11.529 | 5.897 | 1.05/1.05 | 1.04/1.04 | 2.04/2.01 |
| 1,000,000 | distinct | 9 | 10.716 | 12.752 | 10.439 | 5.420 | 0.84/0.85 | 1.03/1.04 | 1.98/1.99 |
| 2,000,000 | 1 | 1 | 1118967.624 | 12.795 | 9.808 | 10.638 | 87450.46/87450.46 | 114091.92/114091.92 | 105186.41/105186.41 |
| 2,000,000 | 3 | 1 | 119015.240 | 57.389 | 41.026 | 11.622 | 2073.85/2073.85 | 2900.98/2900.98 | 10240.18/10240.18 |
| 2,000,000 | 11 | 8 | 8077.587 | 21.741 | 18.173 | 10.779 | 371.54/237.50 | 444.49/371.14 | 749.38/747.76 |
| 2,000,000 | 101 | 9 | 115.369 | 27.762 | 25.088 | 13.133 | 4.16/4.26 | 4.60/4.69 | 8.78/8.99 |
| 2,000,000 | 1001 | 9 | 25.662 | 29.626 | 24.218 | 12.817 | 0.87/0.87 | 1.06/1.06 | 2.00/2.00 |
| 2,000,000 | distinct | 9 | 27.462 | 27.189 | 26.917 | 13.823 | 1.01/1.01 | 1.02/1.02 | 1.99/1.99 |

`repeat_bl` is slower than `three_way` on all-equal samples up to 1,000,000: two passes (a partition, then the gathering pass) against one. The 1,000,000 all-equal cell, 10.174 ms against 4.189, rests on one round, and the same arm took 10.638 ms at 2,000,000.

## Run 6: all four arms, sorted input and random order again

`make bench-median_candidates` with the sorted table added, four arms, 60 s per cell. Times are best ms; the arm then named `current` was `previous`, and `repeat` was still a copy in the benchmark. Sorted input: distinct values `0, ..., n - 1` already in order.

| n | order | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | ascending | 9 | 0.005 | 0.006 | 0.005 | 0.008 | 0.88/0.84 | 1.00/0.98 | 0.71/0.73 |
| 1,000 | descending | 9 | 0.005 | 0.005 | 0.005 | 0.007 | 0.94/1.00 | 1.07/0.90 | 0.75/0.79 |
| 10,000 | ascending | 9 | 0.051 | 0.052 | 0.049 | 0.071 | 0.98/0.99 | 1.04/1.02 | 0.71/0.65 |
| 10,000 | descending | 9 | 0.046 | 0.051 | 0.041 | 0.048 | 0.90/1.02 | 1.12/1.14 | 0.96/0.95 |
| 100,000 | ascending | 9 | 1.093 | 1.037 | 0.924 | 1.411 | 1.05/1.13 | 1.18/1.15 | 0.77/0.85 |
| 100,000 | descending | 9 | 0.691 | 0.722 | 0.411 | 0.447 | 0.96/0.92 | 1.68/1.28 | 1.55/1.40 |
| 500,000 | ascending | 9 | 2.780 | 2.721 | 2.550 | 3.556 | 1.02/1.00 | 1.09/1.10 | 0.78/0.70 |
| 500,000 | descending | 9 | 2.316 | 2.522 | 2.220 | 2.424 | 0.92/0.91 | 1.04/0.77 | 0.96/0.70 |
| 1,000,000 | ascending | 9 | 10.650 | 6.207 | 10.369 | 13.859 | 1.72/1.05 | 1.03/1.09 | 0.77/0.77 |
| 1,000,000 | descending | 9 | 8.750 | 9.144 | 8.469 | 9.272 | 0.96/1.03 | 1.03/1.05 | 0.94/0.97 |
| 2,000,000 | ascending | 9 | 22.707 | 22.086 | 21.667 | 24.354 | 1.03/1.02 | 1.05/1.10 | 0.93/0.80 |
| 2,000,000 | descending | 9 | 10.240 | 10.425 | 10.040 | 10.346 | 0.98/0.90 | 1.02/1.03 | 0.99/0.98 |

The same binary then re-measured run 5's samples:

| n | levels | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.261 | 0.005 | 0.005 | 0.005 | 54.15/57.64 | 51.47/57.78 | 50.34/55.29 |
| 1,000 | 3 | 9 | 0.053 | 0.005 | 0.006 | 0.007 | 9.81/6.14 | 9.50/7.01 | 7.29/6.74 |
| 1,000 | 11 | 9 | 0.013 | 0.005 | 0.008 | 0.005 | 2.43/1.75 | 1.52/1.17 | 2.38/1.64 |
| 1,000 | 101 | 9 | 0.010 | 0.007 | 0.010 | 0.006 | 1.53/1.62 | 1.06/1.61 | 1.62/2.70 |
| 1,000 | 1001 | 9 | 0.009 | 0.007 | 0.008 | 0.006 | 1.16/1.28 | 1.08/0.94 | 1.37/0.89 |
| 1,000 | distinct | 9 | 0.009 | 0.008 | 0.007 | 0.006 | 1.13/1.15 | 1.19/1.01 | 1.43/1.34 |
| 10,000 | 1 | 9 | 28.224 | 0.047 | 0.054 | 0.053 | 595.80/629.90 | 522.43/476.94 | 534.40/434.73 |
| 10,000 | 3 | 9 | 3.032 | 0.112 | 0.104 | 0.052 | 26.98/29.22 | 29.04/26.69 | 58.68/65.30 |
| 10,000 | 11 | 9 | 0.414 | 0.161 | 0.164 | 0.069 | 2.57/2.97 | 2.52/2.29 | 5.98/4.57 |
| 10,000 | 101 | 9 | 0.153 | 0.134 | 0.143 | 0.063 | 1.14/1.20 | 1.07/1.07 | 2.44/2.90 |
| 10,000 | 1001 | 9 | 0.158 | 0.138 | 0.150 | 0.062 | 1.15/1.07 | 1.05/0.92 | 2.55/2.35 |
| 10,000 | distinct | 9 | 0.156 | 0.162 | 0.154 | 0.074 | 0.96/1.01 | 1.02/1.20 | 2.11/2.26 |
| 100,000 | 1 | 9 | 2852.590 | 0.498 | 0.503 | 0.574 | 5727.90/4084.06 | 5675.11/4969.47 | 4966.87/4289.41 |
| 100,000 | 3 | 9 | 312.400 | 1.372 | 1.241 | 0.554 | 227.64/204.06 | 251.72/183.79 | 564.41/501.51 |
| 100,000 | 11 | 9 | 23.104 | 1.599 | 1.120 | 0.570 | 14.45/16.92 | 20.63/24.54 | 40.52/44.99 |
| 100,000 | 101 | 9 | 1.983 | 1.891 | 1.612 | 0.649 | 1.05/1.09 | 1.23/1.19 | 3.06/3.26 |
| 100,000 | 1001 | 9 | 1.367 | 1.793 | 1.319 | 0.631 | 0.76/0.73 | 1.04/0.97 | 2.16/2.15 |
| 100,000 | distinct | 9 | 1.415 | 1.762 | 1.272 | 0.722 | 0.80/0.79 | 1.11/1.10 | 1.96/1.92 |
| 500,000 | 1 | 1 | 65348.992 | 2.048 | 2.629 | 2.621 | 31907.80/31907.80 | 24852.59/24852.59 | 24931.77/24931.77 |
| 500,000 | 3 | 9 | 6901.993 | 5.243 | 5.197 | 2.387 | 1316.33/1157.85 | 1328.04/1315.29 | 2891.83/2666.96 |
| 500,000 | 11 | 9 | 481.993 | 6.797 | 6.039 | 2.991 | 70.91/75.44 | 79.81/79.61 | 161.13/78.90 |
| 500,000 | 101 | 9 | 12.268 | 6.025 | 6.180 | 2.826 | 2.04/2.48 | 1.99/2.31 | 4.34/3.81 |
| 500,000 | 1001 | 9 | 5.733 | 7.807 | 5.449 | 2.876 | 0.73/0.72 | 1.05/1.01 | 1.99/1.71 |
| 500,000 | distinct | 9 | 5.201 | 8.121 | 4.948 | 2.803 | 0.64/0.64 | 1.05/0.79 | 1.86/1.10 |
| 1,000,000 | 1 | 1 | 310928.230 | 8.678 | 24.013 | 11.569 | 35829.71/35829.71 | 12948.38/12948.38 | 26876.57/26876.57 |
| 1,000,000 | 3 | 2 | 30147.945 | 13.097 | 11.484 | 5.997 | 2301.82/2261.44 | 2625.26/2623.99 | 5027.18/5058.31 |
| 1,000,000 | 11 | 9 | 2175.797 | 15.573 | 11.882 | 5.716 | 139.72/138.93 | 183.12/163.88 | 380.68/366.51 |
| 1,000,000 | 101 | 9 | 37.166 | 16.992 | 11.906 | 7.182 | 2.19/2.33 | 3.12/3.29 | 5.17/5.62 |
| 1,000,000 | 1001 | 9 | 13.151 | 12.566 | 12.415 | 6.286 | 1.05/0.96 | 1.06/1.05 | 2.09/2.16 |
| 1,000,000 | distinct | 9 | 11.187 | 13.739 | 11.175 | 5.664 | 0.81/0.82 | 1.00/1.01 | 1.98/2.07 |
| 2,000,000 | 1 | 1 | 1184647.908 | 8.479 | 9.880 | 10.898 | 139709.45/139709.45 | 119907.46/119907.46 | 108701.20/108701.20 |
| 2,000,000 | 3 | 1 | 135253.305 | 37.439 | 51.379 | 22.269 | 3612.65/3612.65 | 2632.45/2632.45 | 6073.60/6073.60 |
| 2,000,000 | 11 | 7 | 9062.937 | 24.194 | 22.012 | 11.454 | 374.59/218.16 | 411.72/367.23 | 791.23/707.62 |
| 2,000,000 | 101 | 9 | 129.693 | 30.329 | 26.617 | 14.039 | 4.28/3.74 | 4.87/5.17 | 9.24/9.62 |
| 2,000,000 | 1001 | 9 | 28.264 | 33.367 | 26.511 | 13.862 | 0.85/0.87 | 1.07/1.01 | 2.04/2.03 |
| 2,000,000 | distinct | 9 | 30.101 | 30.264 | 29.734 | 14.797 | 0.99/0.99 | 1.01/1.09 | 2.03/2.17 |

Across runs 4 to 6 (`repeat` a copy, not the shipped build) no cell has `repeat` below `previous` in both the best and the median round. Nine cells have it below 1 in one of the two (lowest 0.77, at 500,000 descending in run 6), each at or above 1 in the other: noise by the rule above. `repeat`'s 24.013 ms on 1,000,000 equal values rests on one round, against 9.880 ms at 2,000,000. `repeat_bl` is below 1 in both columns at every ascending size, and below or at 1 on descending input except at 100,000.

## Run 7: the shipped `stats_median`

`make bench-median_candidates` after the deployment: `repeat` is `stats_median` from `stats.h` as shipped, `previous` a copy of the selection it replaced. Four arms, 60 s per cell, times are best ms. This is the run every claim about the shipped build rests on.

| n | order | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | ascending | 9 | 0.006 | 0.006 | 0.006 | 0.007 | 1.05/0.99 | 0.97/0.95 | 0.89/0.82 |
| 1,000 | descending | 9 | 0.004 | 0.005 | 0.005 | 0.004 | 0.87/0.89 | 0.85/0.88 | 1.02/1.02 |
| 10,000 | ascending | 9 | 0.056 | 0.054 | 0.059 | 0.068 | 1.03/1.05 | 0.94/0.98 | 0.82/0.66 |
| 10,000 | descending | 9 | 0.043 | 0.049 | 0.050 | 0.042 | 0.86/0.86 | 0.85/0.88 | 1.02/1.01 |
| 100,000 | ascending | 9 | 0.578 | 0.538 | 0.580 | 0.654 | 1.07/1.15 | 1.00/0.84 | 0.88/0.84 |
| 100,000 | descending | 9 | 0.418 | 0.492 | 0.507 | 0.416 | 0.85/0.89 | 0.82/0.93 | 1.00/1.10 |
| 500,000 | ascending | 9 | 2.946 | 2.848 | 3.222 | 3.366 | 1.03/0.99 | 0.91/0.89 | 0.88/0.89 |
| 500,000 | descending | 9 | 2.229 | 2.520 | 2.623 | 2.292 | 0.88/0.97 | 0.85/0.92 | 0.97/1.05 |
| 1,000,000 | ascending | 9 | 5.909 | 5.705 | 6.185 | 6.929 | 1.04/1.12 | 0.96/1.00 | 0.85/0.90 |
| 1,000,000 | descending | 9 | 4.633 | 5.165 | 5.615 | 4.746 | 0.90/0.89 | 0.83/0.83 | 0.98/1.00 |
| 2,000,000 | ascending | 9 | 11.969 | 11.514 | 12.374 | 13.889 | 1.04/1.08 | 0.97/0.99 | 0.86/0.88 |
| 2,000,000 | descending | 9 | 9.533 | 10.252 | 11.689 | 9.857 | 0.93/0.90 | 0.82/0.83 | 0.97/0.98 |

| n | levels | rounds | previous | three_way | repeat | repeat_bl | three_way best/med | repeat best/med | repeat_bl best/med |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 9 | 0.490 | 0.005 | 0.006 | 0.005 | 93.84/89.35 | 76.01/75.92 | 99.24/94.80 |
| 1,000 | 3 | 9 | 0.059 | 0.005 | 0.006 | 0.005 | 12.01/11.78 | 9.49/9.48 | 11.40/11.65 |
| 1,000 | 11 | 9 | 0.012 | 0.006 | 0.008 | 0.005 | 2.05/2.07 | 1.48/1.48 | 2.27/2.31 |
| 1,000 | 101 | 9 | 0.009 | 0.007 | 0.008 | 0.006 | 1.40/1.39 | 1.10/1.07 | 1.59/1.58 |
| 1,000 | 1001 | 9 | 0.008 | 0.007 | 0.008 | 0.006 | 1.16/1.12 | 1.08/1.05 | 1.44/1.41 |
| 1,000 | distinct | 9 | 0.008 | 0.007 | 0.008 | 0.006 | 1.16/1.20 | 1.06/1.12 | 1.46/1.51 |
| 10,000 | 1 | 9 | 47.320 | 0.051 | 0.065 | 0.049 | 925.56/914.98 | 729.89/741.00 | 963.65/922.67 |
| 10,000 | 3 | 9 | 5.390 | 0.113 | 0.114 | 0.048 | 47.52/48.75 | 47.29/46.99 | 111.95/107.22 |
| 10,000 | 11 | 9 | 0.562 | 0.153 | 0.153 | 0.063 | 3.68/3.68 | 3.68/3.68 | 8.88/8.93 |
| 10,000 | 101 | 9 | 0.145 | 0.132 | 0.139 | 0.057 | 1.10/1.39 | 1.04/1.36 | 2.56/3.56 |
| 10,000 | 1001 | 9 | 0.153 | 0.136 | 0.143 | 0.059 | 1.12/1.77 | 1.07/1.84 | 2.60/4.62 |
| 10,000 | distinct | 9 | 0.149 | 0.160 | 0.147 | 0.063 | 0.93/1.12 | 1.01/1.22 | 2.37/2.97 |
| 100,000 | 1 | 9 | 4828.486 | 0.512 | 0.619 | 0.506 | 9433.86/8358.80 | 7799.37/5219.93 | 9540.77/5419.12 |
| 100,000 | 3 | 9 | 618.925 | 1.351 | 1.385 | 0.610 | 458.20/449.02 | 446.86/388.79 | 1015.23/1034.30 |
| 100,000 | 11 | 9 | 44.114 | 1.719 | 1.275 | 0.584 | 25.66/26.02 | 34.60/33.65 | 75.50/77.00 |
| 100,000 | 101 | 9 | 2.374 | 2.047 | 1.738 | 0.637 | 1.16/1.20 | 1.37/1.33 | 3.72/3.77 |
| 100,000 | 1001 | 9 | 1.442 | 2.036 | 1.410 | 0.661 | 0.71/0.73 | 1.02/0.96 | 2.18/2.14 |
| 100,000 | distinct | 9 | 1.457 | 1.900 | 1.486 | 0.723 | 0.77/0.77 | 0.98/1.05 | 2.01/2.09 |
| 500,000 | 1 | 1 | 127964.918 | 5.944 | 7.500 | 5.751 | 21527.62/21527.62 | 17061.99/17061.99 | 22252.65/22252.65 |
| 500,000 | 3 | 5 | 13545.343 | 5.739 | 5.846 | 2.447 | 2360.29/2381.47 | 2317.12/2331.47 | 5534.40/5544.99 |
| 500,000 | 11 | 9 | 1046.042 | 7.553 | 6.655 | 3.043 | 138.49/114.58 | 157.18/156.99 | 343.77/346.67 |
| 500,000 | 101 | 9 | 18.207 | 6.570 | 8.098 | 2.917 | 2.77/3.49 | 2.25/3.10 | 6.24/7.76 |
| 500,000 | 1001 | 9 | 6.770 | 9.756 | 7.303 | 3.319 | 0.69/0.77 | 0.93/1.01 | 2.04/2.08 |
| 500,000 | distinct | 9 | 6.488 | 9.702 | 6.117 | 3.124 | 0.67/0.69 | 1.06/1.07 | 2.08/2.18 |
| 1,000,000 | 1 | 1 | 511214.723 | 11.298 | 15.856 | 12.300 | 45247.60/45247.60 | 32242.09/32242.09 | 41562.64/41562.64 |
| 1,000,000 | 3 | 2 | 54585.797 | 11.574 | 11.890 | 5.749 | 4716.11/4777.30 | 4590.97/4633.18 | 9494.27/9614.43 |
| 1,000,000 | 11 | 9 | 4112.094 | 15.407 | 11.880 | 5.447 | 266.89/258.17 | 346.14/348.02 | 754.98/756.27 |
| 1,000,000 | 101 | 9 | 57.323 | 17.061 | 12.357 | 6.689 | 3.36/4.57 | 4.64/6.30 | 8.57/11.67 |
| 1,000,000 | 1001 | 9 | 13.332 | 12.527 | 12.807 | 6.069 | 1.06/0.85 | 1.04/0.93 | 2.20/2.19 |
| 1,000,000 | distinct | 9 | 11.703 | 14.147 | 11.633 | 5.573 | 0.83/0.91 | 1.01/0.87 | 2.10/2.30 |
| 2,000,000 | 1 | 1 | 2024397.495 | 10.541 | 13.218 | 10.894 | 192050.18/192050.18 | 153149.87/153149.87 | 185823.68/185823.68 |
| 2,000,000 | 3 | 1 | 223322.375 | 23.328 | 28.774 | 11.836 | 9573.16/9573.16 | 7761.30/7761.30 | 18867.58/18867.58 |
| 2,000,000 | 11 | 4 | 15981.850 | 23.663 | 20.665 | 11.073 | 675.39/676.04 | 773.37/775.35 | 1443.28/1447.30 |
| 2,000,000 | 101 | 9 | 207.762 | 30.442 | 27.766 | 13.395 | 6.82/8.44 | 7.48/9.20 | 15.51/19.16 |
| 2,000,000 | 1001 | 9 | 29.215 | 32.436 | 26.549 | 13.083 | 0.90/0.90 | 1.10/1.10 | 2.23/2.23 |
| 2,000,000 | distinct | 9 | 30.410 | 29.594 | 29.380 | 14.083 | 1.03/1.03 | 1.04/1.03 | 2.16/2.16 |

Shipped `repeat` against `previous`, this run only:

- Sorted descending: slower at every length, below 1 in both columns, best 0.82 to 0.85 (1,000,000: 4.633 ms against 5.615).
- Sorted ascending: below 1 in both columns at 1,000, 10,000, 500,000 and 2,000,000 (0.89 to 0.99), noise at 100,000 and 1,000,000.
- Distinct values in random order: 0.98 to 1.06 best, no cell below 1 in both columns.
- 1 to 101 distinct values: 1.04 to 153,150 faster; the median of 2,000,000 equal values took 2,024 s with `previous` and 13.2 ms with `repeat`.
- 1,001 distinct values: 0.93 to 1.10 best, no cell below 1 in both columns.

