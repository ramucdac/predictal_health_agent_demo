# Predictal AI Application Health Agent

A demo Streamlit dashboard for monitoring a Pega application using simulated PDC telemetry and LangChain AI analysis.

## Features

- Predictal PROD / QA / DEV selection
- Application health score
- Response time
- Requestors per node
- JVM heap
- CPU
- HTTP 504 errors
- Connector timeouts
- PDC event summary
- PDC event numbers and matching ServiceNow case numbers
- 60-minute charts
- LangChain AI root-cause analysis
- Recommended actions
- Automatic email alerts for Critical/High PDC events that match an earlier same-environment ServiceNow incident

## Run locally

### 1. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI and email alerts

Copy `.env.example` to `.env`:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.6
SMTP_HOST=smtp.your-company.example
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password
ALERT_FROM_EMAIL=predictal-monitor@your-company.example
ALERT_TO_EMAILS=team@example.com,oncall@example.com
ENABLE_EMAIL_ALERTS=false
```

When email delivery is enabled, the app automatically sends an email for a
Critical or High PDC event with matching symptoms in an earlier, resolved
ServiceNow incident for the selected environment. Each event displays a PDC event number and matching ServiceNow
case number(s). Browser response time uses event number `PEGA0001`, and the
connector total-time/timeout event uses `PEGA00020`. The dashboard displays
the email subject, recipients, and message body for matching Critical/High
events. Simulated telemetry is deterministic for reliable demos. SMTP delivery
is disabled by default; set `ENABLE_EMAIL_ALERTS=true` only when you want to
send real email. When enabled, an alert is attempted once per event number and
minute in the current Streamlit session to avoid duplicates on app reruns.
Configure SMTP with your organization's approved relay and team distribution
list. Leave both SMTP credential values empty if the relay does not require
authentication. Do not commit `.env`.

### 4. Start dashboard

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, normally:

http://localhost:8501

## Important

The PDC telemetry in this demo is simulated.

To connect real PDC data, replace:

```python
get_pdc_telemetry()
```

with your approved PDC API client.

Keep deterministic health scoring separate from the LLM. LangChain should explain/correlate telemetry rather than inventing measurements.
