"""
alerts.py
---------
Alerting system. When a transaction's risk_score crosses the configured
threshold (config.ALERT_RISK_SCORE_THRESHOLD), an alert is:
  1. printed to the console,
  2. persisted to the database (alerts table),
  3. optionally emailed (only if ALERT_EMAIL_ENABLED=true and credentials are
     supplied via .env -- never hard-coded).

Fixes vs. original spec: the incomplete smtplib snippet (undefined
sender/recipient/body variables) is replaced with a real, working function
that reads credentials from environment variables only.
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

from src import config
from src.database import insert_alert
from src.utils import get_logger

logger = get_logger(__name__)


def should_alert(risk_score: int) -> bool:
    return risk_score >= config.ALERT_RISK_SCORE_THRESHOLD


def send_console_alert(transaction_id: str, risk_level: str, risk_score: int) -> None:
    print(f"[ALERT] Transaction {transaction_id} flagged {risk_level} risk (score={risk_score})")


def send_email_alert(transaction_id: str, risk_level: str, risk_score: int) -> bool:
    """Send an email alert. Returns True on success, False if disabled/failed."""
    if not config.ALERT_EMAIL_ENABLED:
        logger.info("Email alerts disabled (ALERT_EMAIL_ENABLED=false); skipping email")
        return False

    if not (config.ALERT_EMAIL and config.ALERT_EMAIL_PASSWORD and config.ALERT_RECEIVER):
        logger.warning("Email alert requested but credentials are missing in .env; skipping")
        return False

    subject = f"[Fraud Alert] {risk_level} risk transaction {transaction_id}"
    body = (
        f"Transaction {transaction_id} was flagged as {risk_level} risk "
        f"with a risk score of {risk_score}/100. Please investigate."
    )
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = config.ALERT_EMAIL
    message["To"] = config.ALERT_RECEIVER

    try:
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.ALERT_EMAIL, config.ALERT_EMAIL_PASSWORD)
            server.sendmail(config.ALERT_EMAIL, [config.ALERT_RECEIVER], message.as_string())
        logger.info(f"Email alert sent for transaction {transaction_id}")
        return True
    except Exception as exc:
        logger.error(f"Failed to send email alert: {exc}")
        return False


def raise_alert(transaction_id: str, risk_level: str, risk_score: int) -> dict:
    """Run the full alert pipeline (console + DB + optional email) for a high-risk transaction."""
    message = f"{risk_level} risk transaction detected (score={risk_score})"

    send_console_alert(transaction_id, risk_level, risk_score)
    alert_id = insert_alert(transaction_id, risk_level, message)
    email_sent = send_email_alert(transaction_id, risk_level, risk_score)

    return {"alert_id": alert_id, "transaction_id": transaction_id, "risk_level": risk_level,
            "risk_score": risk_score, "email_sent": email_sent}
