"""Kowalski / Gloria / Melman connection helpers for scope-ml.

Wraps the `penquins.Kowalski` client with scope-ml's token-from-env and
multi-instance configuration conventions. Keeps all external-archive access
in ``scope.surveys``.

Usage
-----
>>> from scope.surveys.kowalski import make_kowalski_client
>>> k = make_kowalski_client(config)
>>> if k is not None:
...     response = k.query({...})
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from penquins import Kowalski


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Override per-host tokens from environment variables in-place.

    Reads:
      - KOWALSKI_INSTANCE_TOKEN
      - GLORIA_INSTANCE_TOKEN
      - MELMAN_INSTANCE_TOKEN
    """
    env_map = {
        "kowalski": "KOWALSKI_INSTANCE_TOKEN",
        "gloria": "GLORIA_INSTANCE_TOKEN",
        "melman": "MELMAN_INSTANCE_TOKEN",
    }
    hosts_cfg = config.get("kowalski", {}).get("hosts", {})
    for host_name, env_name in env_map.items():
        token = os.environ.get(env_name)
        if token is not None and host_name in hosts_cfg:
            hosts_cfg[host_name]["token"] = token


def build_instances(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return a ``{host: instance_kwargs}`` mapping suitable for ``Kowalski(instances=...)``.

    Only hosts with a non-``None`` token are included.
    """
    kowalski_cfg = config["kowalski"]
    hosts_cfg = kowalski_cfg["hosts"]
    hosts = [h for h in hosts_cfg if hosts_cfg[h].get("token") is not None]
    return {
        host: {
            "protocol": kowalski_cfg["protocol"],
            "port": kowalski_cfg["port"],
            "host": f"{host}.caltech.edu",
            "token": hosts_cfg[host]["token"],
        }
        for host in hosts
    }


def make_kowalski_client(
    config: Dict[str, Any],
    *,
    apply_env_overrides: bool = True,
    timeout: Optional[int] = None,
) -> Optional[Kowalski]:
    """Instantiate a ``penquins.Kowalski`` from a scope-ml config dict.

    Parameters
    ----------
    config : dict
        Full scope-ml config (expects ``config['kowalski']`` sub-tree).
    apply_env_overrides : bool, default True
        If True, override per-host tokens from ``*_INSTANCE_TOKEN`` env vars.
    timeout : int, optional
        Override the timeout in seconds. Defaults to ``config['kowalski']['timeout']``.

    Returns
    -------
    Kowalski or None
        ``None`` if no hosts have valid tokens.
    """
    if apply_env_overrides:
        _apply_env_overrides(config)
    instances = build_instances(config)
    if not instances:
        return None
    t = timeout if timeout is not None else config["kowalski"]["timeout"]
    return Kowalski(timeout=t, instances=instances)
