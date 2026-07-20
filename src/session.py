import threading

from curl_cffi import requests

LETTERBOXD_BASE = "https://letterboxd.com"
_REQUEST_DELAY = 1.0
LB_SEM = threading.Semaphore(10)

_session: requests.Session | None = None


def get_curl_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session(impersonate="safari17_0")
    return _session
