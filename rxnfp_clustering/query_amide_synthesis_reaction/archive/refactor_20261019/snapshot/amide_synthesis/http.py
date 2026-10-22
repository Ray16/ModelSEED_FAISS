"""One polite HTTP client for the three public APIs this pipeline calls.

UniProt (04), InterPro (_families) and MetaNetX (_metanetx) each had their own
urlopen call with their own retry policy - or, in two of the three, none at all,
so a single transient failure silently became an empty result indistinguishable
from "no such entry". Centralising it means one place defines the User-Agent,
the backoff and the retry count, and one place distinguishes *failed* from
*empty*.

These are shared public services, not our cluster. Retries back off; callers
rate-limit themselves between calls.
"""
from __future__ import annotations

import time
import os
from urllib.request import Request, urlopen

USER_AGENT = 'ModelSEED-nylon-query'
RETRIES = 4


def get(url: str, timeout: int = 60, retries: int = RETRIES):
    """Fetch `url`. Returns (body_text, headers, failed).

    `failed` is True only when every attempt errored, which is what lets a
    caller tell a network problem apart from a legitimately empty response.
    Never raises: a dead API degrades the run, it does not abort it.
    """
    if os.environ.get('AMIDE_SYNTHESIS_OFFLINE') == '1':
        return None, {}, True
    for attempt in range(retries):
        try:
            request = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read(), dict(response.headers), False
        except Exception as exc:
            if attempt == retries - 1:
                print(f'  ! network failure on {url[:90]}: {exc}', flush=True)
                return None, {}, True
            time.sleep(2 * (attempt + 1))        # linear backoff
    return None, {}, True


def get_text(url: str, timeout: int = 60, retries: int = RETRIES):
    """`get` decoded to str. Returns (text, headers, failed); text is '' on failure."""
    body, headers, failed = get(url, timeout=timeout, retries=retries)
    return (body.decode() if body is not None else ''), headers, failed
