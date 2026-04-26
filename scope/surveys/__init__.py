"""External data-access clients for scope-ml.

Submodules (import-on-demand; none are imported eagerly to avoid forcing
heavy dependencies on consumers that only need one):

rubin    : Rubin LSST TAP / local DP1 clients for forced photometry LCs.
fritz    : Fritz API client for classification/photometry cross-talk.
kowalski : Kowalski / Gloria / Melman connection helper.
wise     : NEOWISE-R TAP and local-parquet clients.

Example
-------
>>> from scope.surveys.wise import make_wise_client, WISELocalClient
"""

__all__ = ["rubin", "fritz", "kowalski", "wise"]
