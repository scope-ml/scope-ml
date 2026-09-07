"""Which model refines which class, and what happens to everything else.

Period finding reports the period at which a light curve repeats.  For most
variables that is the period one wants.  For an eclipsing binary it is not: two
similar eclipses per orbit make the curve repeat every half orbit, so the
periodogram is correct and the reported period is still half the orbital one.
No amount of agreement between algorithms detects this, because they all agree
on the same halved value.

Recovering the orbital period needs a model of the light curve, and the model
has to suit the class, which is why this runs after classification rather than
during feature generation.  Gaia take the same approach: their general
classifier feeds class-specific pipelines that re-derive the period and the
parameters (Eyer et al. 2023; Mowlavi et al. 2023; Clementini et al. 2023).

Classes with no registered model are passed through untouched.  That is
deliberate: this tool rewrites a published quantity, so it should only act
where its model is known to apply and where the result has been validated.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional, Tuple

from . import eclipsing, pulsating


class Model(NamedTuple):
    """A period-refinement model for one family of variables.

    fit_period_models
        Fit every geometry at one trial period; returns a list of candidates.
    select
        Choose among candidates pooled from all trial periods.
    period_range
        Plausible period range for the class, in days.  Trial periods outside
        it are not considered.  This is a prior, and it does real work: for RR
        Lyrae it moves exact agreement from 37.2 to 42.6 per cent on the Gaia
        validation set, because 42 per cent of otherwise-selected periods were
        outside any range an RR Lyrae can occupy.
    global_ranking
        Maps the fraction of unexplained variance to a quality score, following
        Gaia's Eq. 5.  It is well calibrated on our validation data: exact
        agreement rises from 21.6 per cent overall to 62 per cent above a score
        of 0.55, so it is the honest way to say which refined periods to trust.
    characterise
        Optional stage run after a period is chosen, deriving class-specific
        parameters and refining the period continuously.
    """

    fit_period_models: Callable
    select: Callable
    period_range: Tuple[float, float]
    global_ranking: Callable
    characterise: Optional[Callable] = None


#: Period ranges are deliberately wider than the ones Gaia use for candidate
#: selection.  Ours are validated against the Gaia catalogues, so adopting their
#: cuts would be circular: a star excluded by their window is not in our
#: validation set to begin with, and the range could never be shown to be wrong.
REGISTRY: Dict[str, Model] = {
    # Eclipsing binaries and their subtypes.  0.05 to 100 d spans contact
    # systems through long-period detached ones; in practice the refinement
    # only succeeds below about a day on a 257 day baseline, because a longer
    # period leaves too few points inside eclipse to constrain a fit.
    "ECL": Model(
        eclipsing.fit_period_models,
        eclipsing.select,
        (0.05, 100.0),
        eclipsing.global_ranking,
    ),
    "EA": Model(
        eclipsing.fit_period_models,
        eclipsing.select,
        (0.05, 100.0),
        eclipsing.global_ranking,
    ),
    "EB": Model(
        eclipsing.fit_period_models,
        eclipsing.select,
        (0.05, 100.0),
        eclipsing.global_ranking,
    ),
    "EW": Model(
        eclipsing.fit_period_models,
        eclipsing.select,
        (0.05, 100.0),
        eclipsing.global_ranking,
    ),
    "WUMA": Model(
        eclipsing.fit_period_models,
        eclipsing.select,
        (0.05, 100.0),
        eclipsing.global_ranking,
    ),
    # RR Lyrae.  RRc reach down to about 0.2 d and metal-poor RRab up to about
    # 1.2 d, so this has margin at both ends.
    "RR": Model(
        pulsating.fit_period_models,
        pulsating.select,
        (0.1, 1.5),
        pulsating.global_ranking,
        pulsating.characterise,
    ),
    "RRAB": Model(
        pulsating.fit_period_models,
        pulsating.select,
        (0.2, 1.5),
        pulsating.global_ranking,
        pulsating.characterise,
    ),
    "RRC": Model(
        pulsating.fit_period_models,
        pulsating.select,
        (0.1, 0.6),
        pulsating.global_ranking,
        pulsating.characterise,
    ),
    # Cepheids share the Fourier model but occupy a different period regime.
    "CEP": Model(
        pulsating.fit_period_models,
        pulsating.select,
        (0.5, 150.0),
        pulsating.global_ranking,
        pulsating.characterise,
    ),
    "CEPH": Model(
        pulsating.fit_period_models,
        pulsating.select,
        (0.5, 150.0),
        pulsating.global_ranking,
        pulsating.characterise,
    ),
}


def lookup(class_name: Optional[str]) -> Optional[Model]:
    """Model for a class label, or None if the class has no registered model.

    Matching is case insensitive.  A None or empty label returns None, so an
    unclassified source is passed through rather than guessed at.
    """
    if not class_name:
        return None
    return REGISTRY.get(str(class_name).strip().upper())


def supported() -> Tuple[str, ...]:
    """Class labels that have a model, for help text and validation."""
    return tuple(sorted(REGISTRY))
