import os
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

st.set_page_config(
    page_title="Predictal AI Health Monitor",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Demo telemetry
# Replace get_pdc_telemetry() with your real PDC API later.
# -----------------------------
def get_pdc_telemetry(environment: str, minutes: int = 60) -> pd.DataFrame:
    now = datetime.now()
    rows = []

    # Different baseline by environment
    base = {
        "PREDICTAL-PROD": 1.0,
        "PREDICTAL-QA": 0.75,
        "PREDICTAL-DEV": 0.55,
    }.get(environment, 1.0)

    for i in range(minutes):
        ts = now - timedelta(minutes=minutes - 1 - i)

        # Create a realistic degradation pattern near the end.
        degradation = max(0, (i - 38) / 22)

        response = (
            900
            + random.randint(-120, 180)
            + int(2500 * degradation * base)
        )

        requestors = (
            230
            + random.randint(-25, 35)
            + int(180 * degradation * base)
        )

        heap = (
            58
            + random.randint(-4, 5)
            + int(25 * degradation * base)
        )

        cpu = (
            48
            + random.randint(-8, 8)
            + int(32 * degradation * base)
        )

        error_rate = max(
            0.1,
            round(
                0.4
                + random.uniform(0, 0.7)
                + 3.8 * degradation * base,
                2
            )
        )

        timeout = max(
            0,
            int(random.uniform(0, 3) + 30 * degradation * base)
        )

        errors_504 = max(
            0,
            int(random.uniform(0, 3) + 35 * degradation * base)
        )

        alerts = 0
        if response > 2500:
            alerts += 1
        if errors_504 > 20:
            alerts += 1
        if heap > 80:
            alerts += 1

        rows.append({
            "timestamp": ts,
            "response_time_ms": response,
            "requestors_per_node": requestors,
            "heap_percent": min(heap, 98),
            "cpu_percent": min(cpu, 99),
            "error_rate_percent": error_rate,
            "connector_timeouts": timeout,
            "http_504": errors_504,
            "pdc_alerts": alerts,
        })

    return pd.DataFrame(rows)


def calculate_health_score(df: pd.DataFrame):
    latest = df.iloc[-1]
    score = 100
    problems = []

    if latest.response_time_ms > 5000:
        score -= 20
        problems.append("Critical response time")
    elif latest.response_time_ms > 2000:
        score -= 10
        problems.append("High response time")

    if latest.error_rate_percent > 5:
        score -= 20
        problems.append("Critical error rate")
    elif latest.error_rate_percent > 2:
        score -= 10
        problems.append("Elevated error rate")

    if latest.requestors_per_node > 350:
        score -= 15
        problems.append("Requestors per node above threshold")

    if latest.http_504 > 20:
        score -= 15
        problems.append("HTTP 504 spike")

    if latest.connector_timeouts > 20:
        score -= 10
        problems.append("Connector timeouts")

    if latest.heap_percent > 90:
        score -= 15
        problems.append("Critical JVM heap")
    elif latest.heap_percent > 80:
        score -= 8
        problems.append("High JVM heap")

    score = max(0, score)

    if score >= 80:
        status = "HEALTHY"
    elif score >= 60:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return score, status, problems


# -----------------------------
# LangChain tools
# -----------------------------
@tool
def get_current_pega_metrics(environment: str) -> dict:
    """Return the latest simulated Pega/PDC metrics."""
    df = get_pdc_telemetry(environment)
    row = df.iloc[-1]

    return {
        "environment": environment,
        "response_time_ms": int(row.response_time_ms),
        "requestors_per_node": int(row.requestors_per_node),
        "heap_percent": int(row.heap_percent),
        "cpu_percent": int(row.cpu_percent),
        "error_rate_percent": float(row.error_rate_percent),
        "connector_timeouts": int(row.connector_timeouts),
        "http_504": int(row.http_504),
        "pdc_alerts": int(row.pdc_alerts),
    }


@tool
def calculate_pega_health(
    response_time_ms: int,
    requestors_per_node: int,
    heap_percent: int,
    error_rate_percent: float,
    connector_timeouts: int,
    http_504: int,
) -> dict:
    """Calculate a deterministic Pega health score."""
    score = 100

    if response_time_ms > 5000:
        score -= 20
    elif response_time_ms > 2000:
        score -= 10

    if error_rate_percent > 5:
        score -= 20
    elif error_rate_percent > 2:
        score -= 10

    if requestors_per_node > 350:
        score -= 15

    if http_504 > 20:
        score -= 15

    if connector_timeouts > 20:
        score -= 10

    if heap_percent > 90:
        score -= 15
    elif heap_percent > 80:
        score -= 8

    score = max(score, 0)

    status = (
        "HEALTHY" if score >= 80
        else "WARNING" if score >= 60
        else "CRITICAL"
    )

    return {"score": score, "status": status}


SYSTEM_PROMPT = """
You are the Predictal Pega Application Health AI Agent.

Analyze Pega/PDC telemetry and provide an operational assessment.

Important:
- Use the supplied metrics as evidence.
- Do not invent telemetry.
- Distinguish probable root cause from confirmed root cause.
- Correlate metrics before recommending action.

Pay special attention to:
1. Response time
2. Requestors per node
3. JVM heap
4. CPU
5. Error rate
6. HTTP 504 errors
7. Connector timeouts
8. PDC alerts

Useful correlations:
- HTTP 504 + connector timeouts -> possible downstream API latency.
- High requestors + high response time -> possible capacity/requestor saturation.
- High heap + high response time -> possible JVM/GC pressure.
- High CPU + high response time -> possible compute saturation.

Return:
HEALTH SUMMARY
PROBABLE ROOT CAUSE
EVIDENCE
RECOMMENDED ACTIONS
RISK
"""


def run_ai_analysis(environment: str, metrics: dict) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "AI analysis is disabled because OPENAI_API_KEY is not configured.\n\n"
            "Demo rule-based assessment:\n"
            f"- Response time: {metrics['response_time_ms']} ms\n"
            f"- Requestors/node: {metrics['requestors_per_node']}\n"
            f"- Heap: {metrics['heap_percent']}%\n"
            f"- HTTP 504: {metrics['http_504']}\n"
            f"- Connector timeouts: {metrics['connector_timeouts']}\n"
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=[get_current_pega_metrics, calculate_pega_health],
        system_prompt=SYSTEM_PROMPT
    )

    prompt = f"""
Analyze {environment} using these current PDC metrics:

{metrics}

Provide the health assessment and recommended actions.
"""

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })

    return result["messages"][-1].content


# -----------------------------
# Dashboard
# -----------------------------
st.title("🩺 Predictal AI Application Health")
st.caption("Pega Application Monitoring • PDC + LangChain Demo")

with st.sidebar:
    st.header("Monitoring")
    environment = st.selectbox(
        "Environment",
        ["PREDICTAL-PROD", "PREDICTAL-QA", "PREDICTAL-DEV"]
    )

    refresh = st.button("🔄 Refresh telemetry", use_container_width=True)

    st.divider()
    st.write("**Demo thresholds**")
    st.write("Requestors/node: 350")
    st.write("Response warning: 2 sec")
    st.write("Heap warning: 80%")
    st.write("504 warning: 20")

df = get_pdc_telemetry(environment)

score, status, problems = calculate_health_score(df)

latest = df.iloc[-1]

# Top metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Application Health",
        f"{score}/100",
        status
    )

with c2:
    st.metric(
        "Response Time",
        f"{latest.response_time_ms/1000:.2f} sec"
    )

with c3:
    st.metric(
        "Requestors / Node",
        int(latest.requestors_per_node)
    )

with c4:
    st.metric(
        "Error Rate",
        f"{latest.error_rate_percent:.2f}%"
    )

c5, c6, c7, c8 = st.columns(4)

with c5:
    st.metric("JVM Heap", f"{latest.heap_percent}%")

with c6:
    st.metric("CPU", f"{latest.cpu_percent}%")

with c7:
    st.metric("HTTP 504", int(latest.http_504))

with c8:
    st.metric("Connector Timeouts", int(latest.connector_timeouts))

st.divider()

# Status banner
if status == "HEALTHY":
    st.success(f"🟢 Predictal is HEALTHY — {score}/100")
elif status == "WARNING":
    st.warning(f"🟠 Predictal requires attention — {score}/100")
else:
    st.error(f"🔴 Predictal is CRITICAL — {score}/100")

# Charts
st.subheader("📈 Performance — Last 60 Minutes")

left, right = st.columns(2)

with left:
    st.write("Response Time")
    chart = df.set_index("timestamp")[["response_time_ms"]]
    st.line_chart(chart)

with right:
    st.write("Requestors / Node")
    chart = df.set_index("timestamp")[["requestors_per_node"]]
    st.line_chart(chart)

left, right = st.columns(2)

with left:
    st.write("JVM Heap / CPU")
    chart = df.set_index("timestamp")[["heap_percent", "cpu_percent"]]
    st.line_chart(chart)

with right:
    st.write("Errors / Connector Timeouts")
    chart = df.set_index("timestamp")[[
        "http_504",
        "connector_timeouts"
    ]]
    st.line_chart(chart)

# PDC alerts
st.subheader("🚨 PDC Event Summary")

events = []

if latest.http_504 > 20:
    events.append({
        "Time": latest.timestamp.strftime("%H:%M"),
        "Event": "HTTP 504 spike",
        "Source": "External API / Connector",
        "Severity": "CRITICAL"
    })

if latest.connector_timeouts > 20:
    events.append({
        "Time": latest.timestamp.strftime("%H:%M"),
        "Event": "Connector timeout",
        "Source": "Pega Connector",
        "Severity": "HIGH"
    })

if latest.requestors_per_node > 350:
    events.append({
        "Time": latest.timestamp.strftime("%H:%M"),
        "Event": "High requestors",
        "Source": "Pega Web Node",
        "Severity": "WARNING"
    })

if latest.heap_percent > 80:
    events.append({
        "Time": latest.timestamp.strftime("%H:%M"),
        "Event": "High JVM heap",
        "Source": "Pega JVM",
        "Severity": "WARNING"
    })

if events:
    st.dataframe(
        pd.DataFrame(events),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No significant PDC events detected.")

# AI
st.divider()
st.subheader("🤖 LangChain AI Analysis")

metrics = {
    "environment": environment,
    "response_time_ms": int(latest.response_time_ms),
    "requestors_per_node": int(latest.requestors_per_node),
    "heap_percent": int(latest.heap_percent),
    "cpu_percent": int(latest.cpu_percent),
    "error_rate_percent": float(latest.error_rate_percent),
    "connector_timeouts": int(latest.connector_timeouts),
    "http_504": int(latest.http_504),
    "pdc_alerts": int(latest.pdc_alerts),
}

if st.button("🧠 Analyze with LangChain", type="primary"):
    with st.spinner("LangChain is analyzing Predictal telemetry..."):
        analysis = run_ai_analysis(environment, metrics)
    st.markdown(analysis)

st.divider()

st.caption(
    "Demo only: telemetry is simulated. Replace get_pdc_telemetry() "
    "with your approved PDC API integration before production use."
)
