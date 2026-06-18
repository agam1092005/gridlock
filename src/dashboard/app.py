import streamlit as st
import asyncio
import websockets
import json
import threading
import time
import pandas as pd
import pydeck as pdk
from datetime import datetime, timezone

st.set_page_config(page_title="Gridlock 2.0 Live", layout="wide")

# Persistent state
if "incidents" not in st.session_state:
    st.session_state.incidents = {}

st.markdown(
    """
    <style>
    /* Hide Streamlit header anchor links */
    .st-emotion-cache-11jqaew.e101o0h10 a {
        display: none !important;
    }
    a.header-anchor {
        display: none !important;
    }
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

import os


@st.cache_resource
def get_incident_store():
    return {}


@st.cache_resource
def get_latency_store():
    return []


def format_duration(minutes):
    if not minutes:
        return "0 min"
    try:
        minutes = int(float(minutes))
    except (ValueError, TypeError):
        return "0 min"

    if minutes == 0:
        return "0 min"

    days = minutes // 1440
    hrs = (minutes % 1440) // 60
    mins = minutes % 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hrs > 0:
        parts.append(f"{hrs} hr")
    if mins > 0 or not parts:
        parts.append(f"{mins} min")
    return " ".join(parts)


# Global variable to store incidents from the background thread
GLOBAL_INCIDENTS = get_incident_store()
GLOBAL_LATENCIES = get_latency_store()


# Background thread for WebSocket connection
def websocket_thread():
    async def listen_ws():
        api_host = os.environ.get("API_HOST", "localhost")
        uri = f"ws://{api_host}:8000/ws/live"
        while True:
            try:
                print(f"Connecting to WS at {uri}", flush=True)
                async with websockets.connect(uri, ping_interval=None) as ws:
                    print("Connected to WebSocket.", flush=True)
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("type") == "incident_update":
                            print(f"Received incident update: {data['incident_id']}", flush=True)
                            # Store to global dictionary instead of session state
                            GLOBAL_INCIDENTS[data["incident_id"]] = data

                            # Use backend-stamped processing time — avoids server/client
                            # clock-skew that makes (receive_time - sent_time) unreliable.
                            try:
                                processing_ms = data.get("api_process_time_ms")
                                if processing_ms is not None:
                                    GLOBAL_LATENCIES.append(float(processing_ms))
                                    # Keep last 50 for moving average
                                    if len(GLOBAL_LATENCIES) > 50:
                                        GLOBAL_LATENCIES.pop(0)
                            except Exception as e:
                                print(f"Latency tracking error: {e}", flush=True)
            except Exception as e:
                print(f"WebSocket Error: {e}", flush=True)
                import asyncio

                await asyncio.sleep(2)  # reconnect delay

    asyncio.run(listen_ws())


@st.cache_resource
def start_ws():
    t = threading.Thread(target=websocket_thread, daemon=True)
    t.start()
    return True


start_ws()

st.title("🚦 Gridlock 2.0 - Live Dashboard", anchor=False)

tab1, tab_news, tab2, tab3 = st.tabs(
    ["Live Map", "News Feed", "Playbook & Incidents", "System Status"]
)

with tab1:
    col1, col2 = st.columns([3, 1], vertical_alignment="center")
    with col1:
        st.markdown(
            "<h3 style='margin: 0; padding-bottom: 0;'>Real-Time Incident Map</h3>",
            unsafe_allow_html=True,
        )
    with col2:
        show_module_b = st.toggle("Show Spatial-Temporal Graph Congestion", value=False)
        if show_module_b:
            mitigation_mode = st.radio("Forecast Mode", options=["Live/Unmitigated", "With AI Diversion Playbook"])
        else:
            mitigation_mode = "Live/Unmitigated"

    try:
        from st_theme import st_theme

        theme = st_theme()
        is_dark = True
        if theme and theme.get("base") == "light":
            is_dark = False
    except ImportError:
        is_dark = True

    if not is_dark:
        bg_color, text_color, border_color = "#FFFFFF", "#000000", "#CCCCCC"
        faded_color = "#888888"
    else:
        bg_color, text_color, border_color = "#262730", "#FAFAFA", "#444444"
        faded_color = "#CCCCCC"

    # Map real data
    data = []
    heatmap_data = []
    for inc_id, inc in list(GLOBAL_INCIDENTS.items()):
        data.append(
            {
                "incident_id": inc_id,
                "lat": inc.get("location", {}).get("latitude", 37.7749),
                "lon": inc.get("location", {}).get("longitude", -122.4194),
                "severity": inc.get("severity_score", 0),
                "severity_display": f"{inc.get('severity_score', 0):.3f}%",
                "color": [255, 0, 0]
                if inc.get("severity_score", 0) >= 70
                else ([255, 165, 0] if inc.get("severity_score", 0) >= 50 else [0, 255, 0]),
                "type": str(inc.get("incident_type", "Unknown")).replace("_", " ").title(),
                "desc": str(inc.get("description", "No description"))[:100]
                + ("..." if len(str(inc.get("description", ""))) > 100 else ""),
                "duration": inc.get("duration_estimate", 0),
                "duration_display": format_duration(inc.get("duration_estimate", 0)),
                "address": inc.get("metadata", {}).get("address", "Unknown Location")
                if isinstance(inc.get("metadata"), dict)
                else "Unknown Location",
            }
        )

        mod_b = inc.get("module_b_geojson", {})
        if mod_b and "features" in mod_b:
            for feat in mod_b["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [0, 0])
                weight = float(feat.get("properties", {}).get("weight", 0))
                
                if mitigation_mode == "With AI Diversion Playbook":
                    weight *= 0.35
                    
                if weight > 0.05:
                    heatmap_data.append({"lon": coords[0], "lat": coords[1], "weight": weight})

    df = pd.DataFrame(data)
    df_heatmap = pd.DataFrame(heatmap_data)
    layers = []

    if show_module_b and not df_heatmap.empty:
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            df_heatmap,
            opacity=0.6,
            get_position="[lon, lat]",
            get_weight="weight",
            radius_pixels=60,
            intensity=1.5,
            threshold=0.05,
        )
        layers.append(heatmap_layer)

    if not df.empty:
        layer = pdk.Layer(
            "ScatterplotLayer",
            df,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=6,
            radius_min_pixels=10,
            radius_max_pixels=100,
            line_width_min_pixels=1,
            get_position="[lon, lat]",
            get_radius="severity",
            get_fill_color="color",
            get_line_color=[0, 0, 0],
        )
        layers.append(layer)

        # Use static center so PyDeck doesn't reset the user's zoom/pan on every refresh
        view_state = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=11, pitch=50)
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                layers=layers,
                initial_view_state=view_state,
                tooltip={
                    "html": "<div style='background: var(--background-color, #262730); color: var(--text-color, #FAFAFA); border: 1px solid var(--secondary-background-color, #444444); padding: 12px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: -10px;'>"
                    f"<b style='font-size: 1.1em;'>{{type}}</b><br/>"
                    f"<hr style='margin: 8px 0; border: 0; border-top: 1px solid var(--secondary-background-color, {border_color});'/>"
                    f"<b>Severity:</b> {{severity_display}}<br/>"
                    f"<b>Est. Duration:</b> {{duration_display}}<br/>"
                    f"<b>Address:</b> {{address}}<br/><br/>"
                    f"<i style='color: {faded_color};'>{{desc}}</i>"
                    f"</div>",
                    "style": {
                        "padding": "0px",
                        "background": "transparent",
                        "boxShadow": "none",
                        "border": "none",
                    },
                },
            )
        )
    else:
        st.info("Waiting for incoming incidents via WebSocket...")

with tab2:
    st.header("Active Incidents & Playbooks", anchor=False)
    if GLOBAL_INCIDENTS:
        df_incidents = pd.DataFrame(list(GLOBAL_INCIDENTS.values()))
        if not df_incidents.empty:
            if "severity_score" in df_incidents.columns:
                df_incidents["severity_score"] = df_incidents["severity_score"].apply(
                    lambda x: f"{x:.3f}%" if pd.notnull(x) else x
                )
            if "duration_estimate" in df_incidents.columns:
                df_incidents["duration_estimate"] = df_incidents["duration_estimate"].apply(
                    lambda x: format_duration(x) if pd.notnull(x) else x
                )
        st.dataframe(
            df_incidents.drop(columns=["incident_id"], errors="ignore"), use_container_width=True
        )

        def format_incident_label(inc_id):
            inc = GLOBAL_INCIDENTS.get(inc_id, {})
            type_str = str(inc.get("incident_type", "Unknown")).replace("_", " ").title()
            addr_str = (
                inc.get("metadata", {}).get("address", "Unknown Location")
                if isinstance(inc.get("metadata"), dict)
                else "Unknown Location"
            )
            return f"{type_str} at {addr_str[:40]}{'...' if len(addr_str) > 40 else ''}"

        selected_id = st.selectbox(
            "Select Incident for Playbook Details",
            df_incidents["incident_id"],
            format_func=format_incident_label,
            key="selected_incident_dropdown",
        )
        st.subheader(f"Playbook: {format_incident_label(selected_id)}", anchor=False)
        inc = GLOBAL_INCIDENTS[selected_id]

        # Display AI metrics — read raw numeric values from GLOBAL_INCIDENTS (not formatted df)
        raw_sev = inc.get("severity_score", 50)
        try:
            raw_sev = float(raw_sev)
        except (TypeError, ValueError):
            raw_sev = 50.0
        st.write(f"**AI Severity:** {raw_sev:.1f}/100")
        st.write(f"**Estimated Duration:** {format_duration(inc.get('duration_estimate', 30))}")
        st.markdown("---")
        playbook = inc.get("playbook", {})
        if not isinstance(playbook, dict):
            playbook = {}

        severity_bucket = playbook.get("severity_bucket")
        if not severity_bucket:
            # Fallback based on severity score
            if raw_sev >= 70:
                severity_bucket = "high_severity"
            elif raw_sev >= 40:
                severity_bucket = "medium_severity"
            else:
                severity_bucket = "low_severity"

        # Pull resources with safe defaults matching the fallback values
        if severity_bucket == "high_severity":
            default_manpower = "1 Inspector, 4 Traffic Constables"
            default_barricading = "10 Heavy Barricades, 5 Cones"
            default_diversion = "Major detour via adjacent arterial roads. Complete block of node."
        elif severity_bucket == "medium_severity":
            default_manpower = "2 Traffic Constables"
            default_barricading = "4 Standard Barricades"
            default_diversion = "Partial lane closure. Route heavy vehicles to alternate paths."
        else:
            default_manpower = "1 Traffic Constable (Monitor only)"
            default_barricading = "None required"
            default_diversion = "No diversion needed."

        manpower = playbook.get("manpower", default_manpower)
        barricading = playbook.get("barricading", default_barricading)
        diversion = playbook.get("diversion", default_diversion)

        if severity_bucket == "high_severity":
            st.error("🚨 HIGH SEVERITY ACTION REQUIRED")
        elif severity_bucket == "medium_severity":
            st.warning("⚠️ MODERATE SEVERITY")
        else:
            st.info("ℹ️ LOW SEVERITY")

        translate_to_kannada = st.toggle("🌐 Translate for Field Officers (Kannada)", value=False)
        
        KANNADA_TRANSLATIONS = {
            "1 Inspector, 4 Traffic Constables": "1 ಇನ್ಸ್‌ಪೆಕ್ಟರ್, 4 ಟ್ರಾಫಿಕ್ ಕಾನ್‌ಸ್ಟೇಬಲ್‌ಗಳು",
            "2 Traffic Constables": "2 ಟ್ರಾಫಿಕ್ ಕಾನ್‌ಸ್ಟೇಬಲ್‌ಗಳು",
            "1 Traffic Constable (Monitor only)": "1 ಟ್ರಾಫಿಕ್ ಕಾನ್‌ಸ್ಟೇಬಲ್ (ಮಾನಿಟರ್ ಮಾತ್ರ)",
            "10 Heavy Barricades, 5 Cones": "10 ಹೆವಿ ಬ್ಯಾರಿಕೇಡ್‌ಗಳು, 5 ಕೋನ್‌ಗಳು",
            "4 Standard Barricades": "4 ಸ್ಟ್ಯಾಂಡರ್ಡ್ ಬ್ಯಾರಿಕೇಡ್‌ಗಳು",
            "None required": "ಯಾವುದೇ ಅಗತ್ಯವಿಲ್ಲ",
            "Major detour via adjacent arterial roads. Complete block of node.": "ಪಕ್ಕದ ರಸ್ತೆಗಳ ಮೂಲಕ ಪ್ರಮುಖ ಬಳಸುದಾರಿ. ಜಂಕ್ಷನ್ ಸಂಪೂರ್ಣ ಬ್ಲಾಕ್.",
            "Partial lane closure. Route heavy vehicles to alternate paths.": "ಭಾಗಶಃ ಲೇನ್ ಮುಚ್ಚುವಿಕೆ. ಭಾರೀ ವಾಹನಗಳನ್ನು ಪರ್ಯಾಯ ಮಾರ್ಗಗಳಿಗೆ ತಿರುಗಿಸಿ.",
            "No diversion needed.": "ಯಾವುದೇ ತಿರುವು ಅಗತ್ಯವಿಲ್ಲ."
        }

        display_manpower = KANNADA_TRANSLATIONS.get(manpower, manpower) if translate_to_kannada else manpower
        display_barricading = KANNADA_TRANSLATIONS.get(barricading, barricading) if translate_to_kannada else barricading
        display_diversion = KANNADA_TRANSLATIONS.get(diversion, diversion) if translate_to_kannada else diversion

        st.write("### AI Recommended Action Plan")
        st.write(f"**Optimal Manpower:** {display_manpower}")
        st.write(f"**Required Barricading:** {display_barricading}")
        st.write(f"**Diversion Strategy:** {display_diversion}")

        import requests

        if st.button("Accept Actions & Dispatch"):
            api_host = os.environ.get("API_HOST", "localhost")
            try:
                # Use environment API key or a default
                api_key = os.environ.get("API_KEY", "test-key-12345")
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "approval_status": "approved",
                    "finalized_manpower": manpower,
                    "finalized_barricading": barricading
                }
                url = f"http://{api_host}:8000/api/incidents/{selected_id}/feedback"
                
                # Execute synchronous POST request
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    st.success("Actions dispatched and feedback logged.")
                else:
                    st.error(f"Failed to dispatch: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Network error: {e}")

        # Display SHAP Explanations
        explanations = inc.get("explanations", {})
        if explanations:
            st.markdown("---")
            st.write("### AI Explainability (SHAP)")
            st.caption("Top factors driving the AI's predictions")

            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.write("**Severity Drivers:**")
                for item in explanations.get("severity_shap", {}).get("top_features", []):
                    st.write(f"- {item.get('name', 'Unknown')}: {item.get('shap_value', 0):.2f}")
            with e_col2:
                st.write("**Duration Drivers:**")
                for item in explanations.get("duration_shap", {}).get("top_features", []):
                    st.write(f"- {item.get('name', 'Unknown')}: {item.get('shap_value', 0):.2f}")

        # Display full metadata
        if inc.get("metadata"):
            with st.expander("View Full Historical/Real-time Metadata"):
                st.json(inc["metadata"])
    else:
        st.write("No active incidents to display.")

with tab3:
    st.header("System Status & Latency Monitoring", anchor=False)
    col1, col2, col3 = st.columns(3)

    ws_status = "Connected" if len(GLOBAL_INCIDENTS) > 0 else "Waiting for data..."

    col1.metric("WebSocket Status", ws_status, "0 errors")
    col2.metric("Total Incidents Tracked", len(GLOBAL_INCIDENTS), "Live")

    # backend-stamped processing time (clock-skew proof)
    if GLOBAL_LATENCIES:
        avg_latency_val = sum(GLOBAL_LATENCIES) / len(GLOBAL_LATENCIES)
        avg_latency = f"{avg_latency_val:.0f} ms"

        # Current trend (difference between latest and rolling average)
        delta_val = GLOBAL_LATENCIES[-1] - avg_latency_val
        delta_str = f"{delta_val:+.0f}ms"
    else:
        avg_latency = "N/A"
        delta_str = "0ms"

    col3.metric(
        "Avg ML Processing Time",
        avg_latency,
        delta_str,
        delta_color="inverse",
        help="Pure backend ML pipeline time (api_process_time_ms) stamped by the server. "
        "Not affected by server/client clock skew.",
    )

    st.write("Recent history playback controls will be integrated here.")

with tab_news:
    st.subheader("Live State News", anchor=False)

    # Extract latest news from any recent incident payload
    latest_news = []
    active_alerts = []
    if GLOBAL_INCIDENTS:
        # Get the latest incident by checking the last inserted
        last_inc = list(GLOBAL_INCIDENTS.values())[-1]
        latest_news = last_inc.get("latest_news", [])
        active_alerts = last_inc.get("active_news_alerts", [])

    if latest_news:
        if active_alerts:
            st.markdown(
                f"🚨 **ACTIVE ALERTS:** {', '.join(active_alerts)} *(AI adjustments applied)*"
            )
        else:
            st.markdown("✅ No critical alerts affecting traffic.")

        st.markdown("---")

        for n in latest_news:
            title = n.get("title", "").replace('"', "&quot;")
            source = n.get("source", "")
            pub_date = n.get("pub_date", "")
            st.markdown(
                f"""
            <div style="border: 1px solid {border_color}; border-radius: 12px; padding: 16px; margin-bottom: 12px; background-color: {bg_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; margin-bottom: 8px; font-size: 1.1em; color: {text_color};">{title}</h4>
                <small style="color: {faded_color}; opacity: 0.8;">{source} &bull; {pub_date}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Awaiting live news sync...")

# Auto-refresh every 2 seconds to show new incidents
time.sleep(2)
st.rerun()
