# Period refinement

Feature generation reports the period at which a light curve **repeats**. For
most variables that is the period you want. For an eclipsing binary it is not:
two similar eclipses per orbit make the curve repeat every half orbit, so the
reported period is half the orbital one.

The period finders are not failing. They report the photometric repetition
period correctly — the catalogue column is simply labelled as *the* period, and
for these stars that is a different physical quantity. Cross-algorithm
agreement cannot detect it either, because every algorithm agrees on the same
halved value:

> On 3,866 DP2 sources matched to Gaia DR3 eclipsing binaries, all five
> algorithms agree on **100%** of them, and `best_agree_period` is exact for
> **0.9%**.

Recovering the orbital period requires fitting a model, and the model has to
match the class. That is why this runs after classification rather than during
feature generation.

## Results

Exact agreement with the Gaia DR3 period, within 2%:

| class | n | before | after |
|---|---|---|---|
| eclipsing binaries | 3,866 | 0.9% | **21.2%** |
| RR Lyrae | 406 | 38.9% | **48.3%** |

For eclipsing binaries that is 813 periods corrected against 26 made worse.

Gaia's `global_ranking` is written out with each period and is well calibrated
on this data, so you can trade completeness for accuracy:

| `period_model_rank` | kept | exact |
|---|---|---|
| all | 100% | 21.2% |
| > 0.45 | 37.5% | 46.0% |
| > 0.50 | 21.3% | 54.2% |
| > 0.55 | 7.6% | 62.0% |

## Usage

```bash
refine-periods \
    --features generated_features/combined.parquet \
    --classes  classifications.parquet \
    --output   combined_refined.parquet \
    --min-cadence-minutes 0
```

Stored trial periods are not sufficient on their own: the models are fitted
to the photometry, so the light curves are fetched again. Point the tool at
them the same way `generate-features-rubin` does, with `rubin.data_path` or
`rubin.dp2_data_path` in `config.yaml`, or with `RUBIN_DATA_PATH` /
`RUBIN_DP2_DATA_PATH` in the environment when running from a job script with
no checkout in the working directory. `RUBIN_RELEASE` selects the release and
defaults to `dp2`.

Only `--survey rubin` is implemented.

The input is never modified. The output is the input table plus four columns:

| column | meaning |
|---|---|
| `period_refined` | the refined period, or the original if not refined |
| `period_was_refined` | whether a model actually ran |
| `period_model` | which geometry won, e.g. `TWOGAUSSIANS` |
| `period_model_rank` | quality score, see the table above |

The fits also write out the shape parameters they derive, which are the
quantities the variability subtypes are actually defined on and are worth
carrying into a later classification pass:

| column | class | meaning |
|---|---|---|
| `refined_eclipse_width` | eclipsing | eclipse width in phase; width against period separates detached from contact |
| `refined_eclipse_separation` | eclipsing | phase between the two eclipses; away from 0.5 means an eccentric orbit |
| `refined_eclipse_depth` | eclipsing | primary depth, mean over bands |
| `refined_eclipse_depth_ratio` | eclipsing | secondary over primary depth |
| `refined_eclipse_depth_ratio_<band>` | eclipsing | the same per band, which constrains the temperature ratio of the two stars — a single-band survey cannot measure this |
| `refined_phi21`, `refined_phi31`, `refined_R21`, `refined_R31`, `refined_amp` | pulsating | Fourier parameters |
| `refined_mode`, `refined_is_rrd` | pulsating | RRab/RRc, and double-mode flag |

Depths are left empty where they are not identifiable rather than filled with a
number that looks like a measurement. Two overlapping eclipses constrain their
combined dip but not the split between them, and a band that never observed one
of the eclipses constrains its depth not at all; either case otherwise produces
values in the thousands of magnitudes. On a 60 source check 54 report a depth
and 6 are withheld.

### Options that matter

`--min-cadence-minutes` **must match what feature generation used**, or the fit
sees different photometry from the period search that produced the candidates.
The pipeline default is 5; the Rubin DIA path uses 0. On a 60 source check the
difference was 18.3% against 21.7% exact, and the two settings agreed on only
23% of individual periods.

`--n-periods` sets how many stored peaks per algorithm are offered (default 50,
all of them, which is what the results above were measured at). 20 gives 21.6%
against 21.2% at 2.5× lower cost, but lowers the ceiling: the true period is
present for 79.2% of eclipsing binaries in the top 20 against 90.6% in the top
50.

`--class-filter ECL` restricts to particular classes. `--chunk`/`--n-chunks`
split the work across a job array.

## Which classes

| class | model | why |
|---|---|---|
| `ECL`, `EA`, `EB`, `EW`, `WUMA` | two-Gaussian family, six geometries | the half-period case |
| `RR`, `RRAB`, `RRC`, `CEP`, `CEPH` | truncated Fourier series | sawtooth, not eclipses |
| everything else | none | passed through untouched |

Classes with no registered model keep their original period and are flagged
`period_was_refined = False`. This tool rewrites a published quantity, so it
only acts where the model is known to apply and the result has been validated.

Long-period variables are deliberately absent: their true period is in the
candidate list only **7.8%** of the time, because it exceeds the 257-day
baseline. No selector can fix that; more epochs will.

## Limits

- **Below about a day.** Accuracy falls from 39% under 0.3 d to 1% above 2 d. A
  5-day binary on a 257-day baseline has too few points inside eclipse to
  constrain a fit.
- **Aliases, not harmonics, are the remaining failure.** Of the periods still
  wrong, 57% are unrelated to the truth rather than a factor of two from it —
  they genuinely fit better than the true period. Offering fewer candidates
  does not fix this.
- **The class gate inherits classifier errors.** A misclassified source gets the
  wrong model and the wrong period prior. `period_model` records which model
  ran, so a later reclassification can identify what to redo.

## How it works

1. Collect the trial periods feature generation already stored, within the
   class's plausible period range. **No new period search is performed.**
2. Clean the light curve exactly as feature generation did (`preprocess.py`).
3. Fit every geometry at every trial period, then pool all candidates before
   selecting. For eclipsing binaries that is Gaia's six geometries — two
   Gaussians, two Gaussians with the ellipsoidal cosine centred on either
   eclipse, one Gaussian, one Gaussian with the cosine, and the cosine alone —
   plus a constant model as a non-variable reference. Pooling matters: it is
   what lets a two-eclipse model at 2P beat a one-eclipse model at P.
4. Among everything within 30 of the best BIC, take the highest-ranked model.
5. For pulsators, refine the period continuously and derive Fourier parameters.

Step 4 is where the half-period case actually resolves, and it is a **prior,
not a measurement**. The ranking runs `TWOGAUSSIANS` (6) down to `ELLIPSOIDAL`
(1), so a two-eclipse solution is preferred over a one-eclipse solution of
comparable quality — and those two differ by a factor of two in period. Gaia
are explicit that this is a deliberate bias in their Sect. 2.3: "circular
systems with two equal-depth eclipses will be favored over eccentric systems
displaying only one eclipse (these two cases differ by a factor of two in their
orbital periods)".

This matters because fit quality alone cannot settle it. A model with two free
eclipses can always fit at least as well as one that forces them equal, so any
test that simply compares the two is decided by the extra freedom rather than
by the data. Four such tests were tried against this validation set — an
asymmetric BIC comparison, a properly nested F-test, a colour depth-ratio, and
a model-free permutation test on the two halves of the folded curve — and all
four landed on the diagonal, with false positives tracking true positives at
every threshold. Choosing among candidate periods with a documented prior
works; measuring the difference between two of them does not.

This follows the Gaia DR3 structure — a general classifier feeding
class-specific pipelines that re-derive period and parameters — adapted in one
respect: Gaia fit the single G band, while this fits all bands at once, sharing
the achromatic eclipse timing and width across bands and letting the depths
vary, since the two stars differ in colour.

Two implementation details worth knowing:

- **The fits are unweighted.** In-eclipse points are fainter and carry larger
  errors, so weighting by `1/σ²` suppresses exactly the signal being measured.
  Gaia are explicit about this choice.
- **Unweighted least squares needs `BIC = n·ln(RSS/n) + k·ln(n)`.** Keeping the
  weighted `χ² + k·ln(n)` form lets the penalty dominate, and the
  fewest-parameter model wins regardless of the data. There is a regression
  test for this.

## Performance

The inner loops are compiled with numba, which is already a scope-ml dependency.
Both models evaluate their fit tens of thousands of times per source — every
geometry at every trial period, and for pulsators again for every bootstrap
resample — so the cost is spread across many small calls rather than
concentrated in one hotspot. That is the case compilation addresses and
micro-optimisation does not: rewriting the array code in pure numpy gained 1.7x,
while compiling the residual evaluation and the small linear solve together
gained 4.2x on top of it.

At the default 50 trial periods per algorithm, measured on Rubin DP2:
**1.4 s/object** for eclipsing binaries and **0.3 s/object** for pulsators,
single core, excluding the light curve fetch.

| scope | n | core-hours |
|---|---|---|
| Gaia validation set (eclipsing) | 3,866 | 1.5 |
| full DP2, ungated | 328,285 | 128 |

Gated to a class the figure is lower in proportion to how much of the sample
that class accounts for, which is not yet known for DP2.

The work is per-source and independent, so `--chunk`/`--n-chunks` parallelise it
across a job array with no coordination.

Both modules fall back to pure Python if numba is unavailable, so they still
import and pass their tests; they are simply much slower. The decorators use
`@njit` rather than a bare `@jit` deliberately — a bare `@jit` silently drops to
object mode when it cannot compile, which would turn a compilation regression
into a quiet fifty-fold slowdown instead of an error.

## References

- Mowlavi et al. 2023, *Gaia DR3: the first Gaia catalogue of eclipsing-binary
  candidates* ([arXiv:2211.00929](https://arxiv.org/abs/2211.00929))
- Clementini et al. 2023, *Gaia DR3: Specific processing and validation of all
  sky RR Lyrae and Cepheid stars — the RR Lyrae sample*
  ([arXiv:2206.06278](https://arxiv.org/abs/2206.06278))
