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


def get_sample_servicenow_incidents(environment: str) -> pd.DataFrame:
    now = datetime.now()
    incidents = [
        {
            "number": "INC1002401",
            "environment": "PREDICTAL-PROD",
            "priority": "1 - Critical",
            "state": "In Progress",
            "symptoms": ["response_time", "http_504", "connector_timeout", "peak_traffic"],
            "short_description": "Slow transactions and intermittent 504 responses",
            "description": (
                "Users report slow case loads and intermittent gateway timeouts. "
                "The issue began during peak traffic."
            ),
            "assignment_group": "Pega Platform Support",
            "opened_at": now - timedelta(hours=1, minutes=18),
        },
        {
            "number": "INC1002402",
            "environment": "PREDICTAL-PROD",
            "priority": "2 - High",
            "state": "New",
            "symptoms": ["connector_timeout", "downstream_api", "http_504"],
            "short_description": "Connector requests timing out to customer API",
            "description": (
                "A customer API connector has intermittent timeouts. "
                "The downstream service has not yet been confirmed as the cause."
            ),
            "assignment_group": "Integration Operations",
            "opened_at": now - timedelta(minutes=42),
        },
        {
            "number": "INC1002403",
            "environment": "PREDICTAL-QA",
            "priority": "3 - Moderate",
            "state": "Assigned",
            "symptoms": ["response_time", "load_test"],
            "short_description": "Elevated response time during regression tests",
            "description": (
                "Response time increased while the regression suite was running. "
                "No customer impact reported."
            ),
            "assignment_group": "Quality Engineering",
            "opened_at": now - timedelta(hours=2, minutes=5),
        },
        {
            "number": "INC1002404",
            "environment": "PREDICTAL-DEV",
            "priority": "4 - Low",
            "state": "Resolved",
            "symptoms": ["heap", "test_data"],
            "short_description": "High heap utilization after test data load",
            "description": (
                "Heap utilization increased during a development data load and "
                "returned to normal after the test completed."
            ),
            "assignment_group": "Development Platform",
            "opened_at": now - timedelta(hours=5),
        },
    ]

    return pd.DataFrame(
        incident
        for incident in incidents
        if incident["environment"] == environment
    )


def find_similar_servicenow_incidents(
    incident: dict,
    limit: int = 3,
) -> pd.DataFrame:
    now = datetime.now()
    history = [
        {
            "number": "INC0998420",
            "environment": "PREDICTAL-PROD",
            "priority": "1 - Critical",
            "short_description": "Case loads slow with repeated gateway timeouts",
            "symptoms": ["response_time", "http_504", "connector_timeout"],
            "root_cause": "Customer profile API latency caused connector retries to queue.",
            "resolution": (
                "Coordinated with the API owner to restore service, then verified "
                "connector latency and 504 rates returned to baseline."
            ),
            "resolved_at": now - timedelta(days=18),
        },
        {
            "number": "INC0997314",
            "environment": "PREDICTAL-PROD",
            "priority": "2 - High",
            "short_description": "Connector calls to customer API exceeded timeout",
            "symptoms": ["connector_timeout", "downstream_api"],
            "root_cause": "Intermittent latency in the downstream customer API.",
            "resolution": (
                "Confirmed recovery with the API owner and replayed failed requests "
                "after validating that duplicate processing was prevented."
            ),
            "resolved_at": now - timedelta(days=31),
        },
        {
            "number": "INC0996208",
            "environment": "PREDICTAL-PROD",
            "priority": "2 - High",
            "short_description": "Elevated case response time during peak volume",
            "symptoms": ["response_time", "peak_traffic"],
            "root_cause": "Requestor queue growth coincided with peak transaction volume.",
            "resolution": (
                "Reduced the queued workload and confirmed response time recovered; "
                "capacity tuning was tracked as a follow-up."
            ),
            "resolved_at": now - timedelta(days=46),
        },
        {
            "number": "INC0995092",
            "environment": "PREDICTAL-QA",
            "priority": "3 - Moderate",
            "short_description": "Slow response during regression test execution",
            "symptoms": ["response_time", "load_test"],
            "root_cause": "Concurrent regression jobs saturated the QA test nodes.",
            "resolution": (
                "Staggered the regression jobs and confirmed response time returned "
                "to the normal test baseline."
            ),
            "resolved_at": now - timedelta(days=12),
        },
        {
            "number": "INC0994187",
            "environment": "PREDICTAL-DEV",
            "priority": "4 - Low",
            "short_description": "Heap utilization increased after importing test data",
            "symptoms": ["heap", "test_data"],
            "root_cause": "A large development data import temporarily increased heap use.",
            "resolution": (
                "Completed the import in smaller batches and verified heap returned "
                "to its normal range after processing."
            ),
            "resolved_at": now - timedelta(days=8),
        },
    ]

    current_symptoms = set(incident.get("symptoms", []))
    matches = []

    for past_incident in history:
        past_symptoms = set(past_incident["symptoms"])
        shared_symptoms = current_symptoms & past_symptoms
        all_symptoms = current_symptoms | past_symptoms
        if not shared_symptoms or not all_symptoms:
            continue

        match = past_incident.copy()
        match["similarity_percent"] = round(
            100 * len(shared_symptoms) / len(all_symptoms)
        )
        match["matched_symptoms"] = sorted(shared_symptoms)
        match["same_environment"] = (
            past_incident["environment"] == incident.get("environment")
        )
        matches.append(match)

    matches.sort(
        key=lambda match: (match["similarity_percent"], match["same_environment"]),
        reverse=True,
    )
    return pd.DataFrame(matches[:limit])


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


def run_ai_analysis(
    environment: str,
    metrics: dict,
    incident: dict | None = None,
    similar_incidents: list[dict] | None = None,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        if incident and metrics["http_504"] > 20 and metrics["connector_timeouts"] > 20:
            correlation = (
                "The reported timeout symptoms align with elevated HTTP 504s and "
                "connector timeouts. Confirm downstream API health and connector logs."
            )
        elif (
            incident
            and metrics["response_time_ms"] > 2000
            and metrics["requestors_per_node"] > 350
        ):
            correlation = (
                "The reported latency may align with elevated response time and "
                "requestor load. Check node capacity and requestor queues."
            )
        else:
            correlation = (
                "The current metrics do not confirm the incident cause. Compare "
                "the affected time window with PDC and application logs."
            )

        incident_summary = ""
        if incident:
            incident_summary = (
                f"\n\nServiceNow sample: {incident['number']} "
                f"({incident['priority']}, {incident['state']})\n"
                f"Reported issue: {incident['short_description']}\n"
                f"Description: {incident['description']}\n"
                f"Initial correlation: {correlation}"
            )

        historical_summary = ""
        if similar_incidents:
            historical_summary = "\n\nSimilar resolved sample incidents and fixes:"
            for past_incident in similar_incidents:
                historical_summary += (
                    f"\n- {past_incident['number']} "
                    f"({past_incident['similarity_percent']}% match): "
                    f"{past_incident['short_description']} "
                    f"Root cause: {past_incident['root_cause']} "
                    f"Resolution: {past_incident['resolution']}"
                )

        return (
            "AI analysis is disabled because OPENAI_API_KEY is not configured.\n\n"
            "Demo rule-based assessment:\n"
            f"- Response time: {metrics['response_time_ms']} ms\n"
            f"- Requestors/node: {metrics['requestors_per_node']}\n"
            f"- Heap: {metrics['heap_percent']}%\n"
            f"- HTTP 504: {metrics['http_504']}\n"
            f"- Connector timeouts: {metrics['connector_timeouts']}\n"
            f"{incident_summary}"
            f"{historical_summary}"
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

Selected ServiceNow sample incident:
{incident or "No incident selected"}

Similar resolved sample incidents and recorded resolutions:
{similar_incidents or "No matching history found"}

Correlate the reported incident symptoms with the supplied telemetry. Treat the
incident description as user-reported information, not a confirmed root cause.
Treat historical sample resolutions as prior examples, not proof of the current
root cause or a guaranteed fix.
Provide the health assessment, evidence, next diagnostic steps, and recommended
actions.
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
        width="stretch",
        hide_index=True
    )
else:
    st.info("No significant PDC events detected.")

# ServiceNow incident samples
st.divider()
st.subheader("🎫 ServiceNow Incident Samples")
st.caption("Sample records for demonstration; no live ServiceNow instance is queried.")

incidents = get_sample_servicenow_incidents(environment)
incident_columns = [
    "number",
    "priority",
    "state",
    "short_description",
    "assignment_group",
    "opened_at",
]
st.dataframe(
    incidents[incident_columns],
    width="stretch",
    hide_index=True,
)

incident_descriptions = incidents.set_index("number")["short_description"].to_dict()
selected_incident_number = st.selectbox(
    "Incident to include in analysis",
    incidents["number"].tolist(),
    format_func=lambda number: (
        f"{number} | {incident_descriptions[number]}"
    ),
)
selected_incident = incidents.loc[
    incidents["number"] == selected_incident_number
].iloc[0].to_dict()

st.subheader("🧩 Similar Resolved Incidents and Fixes")
st.caption(
    "Ranked by shared symptoms. Historical incidents and resolutions below are "
    "illustrative demo data, not live ServiceNow records."
)
similar_incidents = find_similar_servicenow_incidents(selected_incident)

if similar_incidents.empty:
    st.info("No similar historical incidents found for this sample.")
else:
    match_columns = [
        "number",
        "similarity_percent",
        "environment",
        "short_description",
        "matched_symptoms",
        "resolved_at",
    ]
    st.dataframe(
        similar_incidents[match_columns],
        width="stretch",
        hide_index=True,
    )

    for past_incident in similar_incidents.to_dict(orient="records"):
        with st.expander(
            f"{past_incident['number']} · "
            f"{past_incident['similarity_percent']}% symptom match · "
            f"{past_incident['short_description']}"
        ):
            st.write(f"**Probable root cause:** {past_incident['root_cause']}")
            st.write(f"**Recorded resolution (demo):** {past_incident['resolution']}")

# AI
st.divider()
st.subheader("🤖 LangChain Incident Analysis")

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

if st.button("🧠 Analyze telemetry + incident", type="primary"):
    with st.spinner("LangChain is correlating telemetry and the incident..."):
        analysis = run_ai_analysis(
            environment,
            metrics,
            selected_incident,
            similar_incidents.to_dict(orient="records"),
        )
    st.markdown(analysis)

st.divider()

st.caption(
    "Demo only: telemetry is simulated. Replace get_pdc_telemetry() "
    "with your approved PDC API integration before production use."
)
