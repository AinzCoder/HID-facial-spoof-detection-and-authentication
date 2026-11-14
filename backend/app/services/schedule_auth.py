import hmac

from fastapi import Header, HTTPException, Request


def require_campus_schedule_admin(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    configured_token = request.app.state.settings.campus_schedule_admin_token
    if configured_token is None:
        raise HTTPException(
            status_code=503,
            detail="Campus schedule administration is disabled until an admin token is configured.",
        )
    scheme, separator, supplied_token = (authorization or "").partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(supplied_token, configured_token)
    ):
        raise HTTPException(
            status_code=401,
            detail="Valid campus schedule administrator credentials are required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
