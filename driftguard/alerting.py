"""Drift alerting.

When the monitoring zone detects dataset drift it dispatches an alert: a webhook
(e.g. a Slack message, or a GitHub ``repository_dispatch`` that kicks off the
retraining workflow). The default dispatcher is a deterministic mock that records
what would be sent; real Slack/webhook dispatchers post over HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logging_setup import get_logger
from .models import DriftReport

log = get_logger("alerting")


@dataclass
class Alert:
    title: str
    drift_share: float
    drifted_features: list[str]
    detail: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "drift_share": round(self.drift_share, 3),
                "drifted_features": self.drifted_features, "detail": self.detail}


class MockDispatcher:
    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def __call__(self, alert: Alert) -> str:
        self.sent.append(alert)
        log.warning("[alert] %s (%.0f%% features drifted)", alert.title, alert.drift_share * 100)
        return f"alert://{len(self.sent)}"


class WebhookDispatcher:  # pragma: no cover - network
    def __init__(self, url: str) -> None:
        self.url = url

    def __call__(self, alert: Alert) -> str:
        import json
        import urllib.request

        body = json.dumps({"event_type": "drift-detected", "client_payload": alert.to_dict()}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
        return "webhook://posted"


class SlackDispatcher:  # pragma: no cover - network
    def __init__(self, webhook: str) -> None:
        self.webhook = webhook

    def __call__(self, alert: Alert) -> str:
        import json
        import urllib.request

        text = (f":rotating_light: *Data drift detected* — {alert.drift_share*100:.0f}% of features shifted "
                f"({', '.join(alert.drifted_features)}). Triggering retraining.")
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(self.webhook, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
        return "slack://posted"


def alert_from_report(report: DriftReport) -> Alert:
    drifted = [f.feature for f in report.features if f.drifted]
    return Alert(title="Data drift detected — model retraining recommended",
                 drift_share=report.drift_share, drifted_features=drifted,
                 detail=f"{report.n_samples} samples; {len(drifted)}/{len(report.features)} features drifted")


def build_dispatcher(webhook_url: str = "", slack_webhook: str = ""):
    if slack_webhook:
        return SlackDispatcher(slack_webhook)
    if webhook_url:
        return WebhookDispatcher(webhook_url)
    return MockDispatcher()
