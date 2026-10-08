# Predictal AI Health Agent Architecture

PDC
 |
 | REST/API
 v
Python PDC Client
 |
 v
Telemetry Normalization
 |
 +--------------------+
 |                    |
 v                    v
Health Rules       LangChain Agent
 |                    |
 +---------+----------+
           |
           v
      Health Report
           |
           v
      Streamlit UI
           |
     +-----+------+
     |            |
   Charts       AI RCA
