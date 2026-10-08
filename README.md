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
- 60-minute charts
- LangChain AI root-cause analysis
- Recommended actions

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

### 3. Configure OpenAI

Copy `.env.example` to `.env`:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.6
```

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
