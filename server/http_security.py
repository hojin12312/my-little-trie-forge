"""Validate HTTP authority, browser origin and optional API credentials."""

import hmac
from urllib.parse import urlsplit

if __package__:
    from .errors import APIError
else:
    from errors import APIError


def validate_api_key(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("API key must contain only visible ASCII characters")
    return value


def authenticate(headers, key):
    if key is None:
        return
    authorization = headers.get_all("Authorization", [])
    api_keys = headers.get_all("x-api-key", [])
    supplied = None
    if len(authorization) == 1 and not api_keys:
        scheme, separator, value = authorization[0].partition(" ")
        if separator and scheme.lower() == "bearer":
            supplied = value
    elif len(api_keys) == 1 and not authorization:
        supplied = api_keys[0]
    if supplied is None or not hmac.compare_digest(supplied.encode(), key.encode()):
        raise APIError(401, "invalid or missing API key", "authentication_error")


def _authority(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid authority")
    parsed = urlsplit("//" + value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid authority")
    return parsed.hostname.lower().rstrip("."), parsed.port


def validate_headers(headers, allowed_hosts):
    hosts = headers.get_all("Host", [])
    origins = headers.get_all("Origin", [])
    try:
        if len(hosts) != 1 or len(origins) > 1:
            raise ValueError("ambiguous authority")
        host, port = _authority(hosts[0])
        if host not in allowed_hosts:
            raise ValueError("untrusted authority")
        if origins:
            origin = urlsplit(origins[0])
            if origin.scheme not in ("http", "https") or origin.path:
                raise ValueError("invalid origin")
            if origin.query or origin.fragment:
                raise ValueError("invalid origin")
            origin_host, origin_port = _authority(origin.netloc)
            default_port = 443 if origin.scheme == "https" else 80
            if (origin_host, default_port if origin_port is None else origin_port) != (
                host,
                default_port if port is None else port,
            ):
                raise ValueError("cross-origin request")
    except ValueError:
        raise APIError(
            403, "request authority or origin is not allowed", "forbidden"
        ) from None
