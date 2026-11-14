import hmac

from fastapi import Header, HTTPException, Request


def require_offline_sync_auth(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    configured_token = request.app.state.settings.offline_sync_token
    if configured_token is None:
        raise HTTPException(
            status_code=503,
            detail="Offline synchronization is disabled until a sync token is configured.",
        )
    scheme, separator, supplied_token = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied_token, configured_token)
    ):
        raise HTTPException(
            status_code=401,
            detail="Valid offline synchronization credentials are required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
