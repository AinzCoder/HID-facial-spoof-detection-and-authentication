from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.database.models import CampusAccessSchedule
from app.schemas import DecisionOutcome
from app.services.decision import Decision, ReasonCode


_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _step_up(reason_code: ReasonCode, reason: str) -> Decision:
    return Decision(DecisionOutcome.STEP_UP, reason_code, reason)


def apply_campus_schedule(
    biometric_decision: Decision,
    schedule: CampusAccessSchedule | None,
    now: datetime,
) -> Decision:
    if biometric_decision.outcome != DecisionOutcome.ALLOW:
        return biometric_decision

    if schedule is None:
        return _step_up(
            ReasonCode.CAMPUS_SCHEDULE_MISSING,
            "No campus access schedule is configured for this user.",
        )
    if not schedule.enabled:
        return _step_up(
            ReasonCode.CAMPUS_SCHEDULE_DISABLED,
            "The campus access schedule is disabled.",
        )

    try:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Current time must be timezone-aware.")
        if not isinstance(schedule.timezone, str):
            raise ValueError("Schedule timezone is invalid.")
        zone = ZoneInfo(schedule.timezone)
        if (
            not isinstance(schedule.permitted_days, list)
            or not schedule.permitted_days
            or any(day not in _WEEKDAYS for day in schedule.permitted_days)
            or not isinstance(schedule.access_start_time, time)
            or not isinstance(schedule.access_end_time, time)
            or schedule.access_start_time.tzinfo is not None
            or schedule.access_end_time.tzinfo is not None
            or schedule.access_start_time >= schedule.access_end_time
        ):
            raise ValueError("Stored campus access schedule is invalid.")

        local_now = now.astimezone(zone)
        if (
            _WEEKDAYS[local_now.weekday()] in schedule.permitted_days
            and schedule.access_start_time <= local_now.time().replace(tzinfo=None)
            < schedule.access_end_time
        ):
            return biometric_decision
    except (AttributeError, KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        return _step_up(
            ReasonCode.CAMPUS_SCHEDULE_EVALUATION_ERROR,
            "The campus access schedule could not be evaluated.",
        )

    return _step_up(
        ReasonCode.OUTSIDE_ACCESS_SCHEDULE,
        "Current local day or time is outside the permitted campus access schedule.",
    )
