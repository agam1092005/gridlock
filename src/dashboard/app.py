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
if 'incidents' not in st.session_state:
    st.session_state.incidents = {}

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
                print(f"Connecting to WS at {uri}")
                async with websockets.connect(uri) as ws:
                    print("Connected to WebSocket.")
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("type") == "incident_update":
                            print(f"Received incident update: {data['incident_id']}")
                            # Store to global dictionary instead of session state
                            GLOBAL_INCIDENTS[data["incident_id"]] = data
                            
                            try:
                                # Track latency
                                sent_time_str = data.get("timestamp", "").replace("Z", "+00:00")
                                if sent_time_str:
                                    sent_time = datetime.fromisoformat(sent_time_str)
                                    receive_time = datetime.now(timezone.utc)
                                    latency_ms = (receive_time - sent_time).total_seconds() * 1000
                                    GLOBAL_LATENCIES.append(latency_ms)
                                    # Keep last 50 for moving average
                                    if len(GLOBAL_LATENCIES) > 50:
                                        GLOBAL_LATENCIES.pop(0)
                            except Exception as e:
                                print(f"Latency tracking error: {e}")
            except Exception as e:
                print(f"WebSocket Error: {e}")
                time.sleep(2) # reconnect delay
                
    asyncio.run(listen_ws())

@st.cache_resource
def start_ws():
    t = threading.Thread(target=websocket_thread, daemon=True)
    t.start()
    return True

start_ws()

st.title("🚦 Gridlock 2.0 - Live Dashboard", anchor=False)

tab1, tab_news, tab2, tab3 = st.tabs(["Live Map", "News Feed", "Playbook & Incidents", "System Status"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Real-Time Incident Map", anchor=False)
    with col2:
        show_module_b = st.toggle("Show Congestion Prediction", value=True)
    
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
    for inc_id, inc in GLOBAL_INCIDENTS.items():
        data.append({
            "incident_id": inc_id,
            "lat": inc.get("location", {}).get("latitude", 37.7749),
            "lon": inc.get("location", {}).get("longitude", -122.4194),
            "severity": inc.get("severity_score", 0),
            "severity_display": f"{inc.get('severity_score', 0):.3f}%",
            "color": [255, 0, 0] if inc.get("severity_score", 0) >= 70 else ([255, 165, 0] if inc.get("severity_score", 0) >= 50 else [0, 255, 0]),
            "type": str(inc.get("incident_type", "Unknown")).replace("_", " ").title(),
            "desc": str(inc.get("description", "No description"))[:100] + ("..." if len(str(inc.get("description", ""))) > 100 else ""),
            "duration": inc.get("duration_estimate", 0),
            "duration_display": format_duration(inc.get("duration_estimate", 0)),
            "address": inc.get("metadata", {}).get("address", "Unknown Location") if isinstance(inc.get("metadata"), dict) else "Unknown Location"
        })
        
        mod_b = inc.get("module_b_geojson", {})
        if mod_b and "features" in mod_b:
            for feat in mod_b["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [0, 0])
                weight = feat.get("properties", {}).get("weight", 0)
                if weight > 0:
                    heatmap_data.append({
                        "lon": coords[0],
                        "lat": coords[1],
                        "weight": weight
                    })
        
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
            threshold=0.05
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
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": f"<div style='background: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 12px; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin: -10px;'>"
                        f"<b style='font-size: 1.1em;'>{{type}}</b><br/>"
                        f"<hr style='margin: 8px 0; border: 0; border-top: 1px solid {border_color};'/>"
                        f"<b>Severity:</b> {{severity_display}}<br/>"
                        f"<b>Est. Duration:</b> {{duration_display}}<br/>"
                        f"<b>Address:</b> {{address}}<br/><br/>"
                        f"<i style='color: {faded_color};'>{{desc}}</i>"
                        f"</div>",
                "style": {
                    "padding": "0px",
                    "background": "transparent",
                    "boxShadow": "none",
                    "border": "none"
                }
            }
        ))
    else:
        st.info("Waiting for incoming incidents via WebSocket...")

with tab2:
    st.header("Active Incidents & Playbooks", anchor=False)
    if GLOBAL_INCIDENTS:
        df_incidents = pd.DataFrame(list(GLOBAL_INCIDENTS.values()))
        if not df_incidents.empty:
            if "severity_score" in df_incidents.columns:
                df_incidents["severity_score"] = df_incidents["severity_score"].apply(lambda x: f"{x:.3f}%" if pd.notnull(x) else x)
            if "duration_estimate" in df_incidents.columns:
                df_incidents["duration_estimate"] = df_incidents["duration_estimate"].apply(lambda x: format_duration(x) if pd.notnull(x) else x)
        st.dataframe(df_incidents)
        
        selected_id = st.selectbox(
            "Select Incident for Playbook Details", 
            df_incidents["incident_id"],
            key="selected_incident_dropdown"
        )
        st.subheader(f"Playbook: {selected_id}", anchor=False)
        inc = GLOBAL_INCIDENTS[selected_id]
        if inc.get("severity_score", 0) >= 70:
            st.error("🚨 HIGH SEVERITY ACTION REQUIRED")
            manpower = "1 Inspector, 4 Traffic Constables"
            barricading = "10 Heavy Barricades, 5 Cones"
            diversion = "Major detour via adjacent arterial roads. Complete block of node."
        elif inc.get("severity_score", 0) >= 40:
            st.warning("⚠️ MODERATE SEVERITY")
            manpower = "2 Traffic Constables"
            barricading = "4 Standard Barricades"
            diversion = "Partial lane closure. Route heavy vehicles to alternate paths."
        else:
            st.info("ℹ️ LOW SEVERITY")
            manpower = "1 Traffic Constable (Monitor only)"
            barricading = "None required"
            diversion = "No diversion needed."
            
        st.write("### AI Recommended Action Plan")
        st.write(f"**Optimal Manpower:** {manpower}")
        st.write(f"**Required Barricading:** {barricading}")
        st.write(f"**Diversion Strategy:** {diversion}")
        
        if st.button("Accept Actions & Dispatch"):
            st.success("Actions dispatched and feedback logged.")
            
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
    
    # Dynamic real-time latency calculation
    if GLOBAL_LATENCIES:
        avg_latency_val = sum(GLOBAL_LATENCIES) / len(GLOBAL_LATENCIES)
        avg_latency = f"{avg_latency_val:.0f} ms"
        
        # Current trend (difference between latest latency and average)
        delta_val = GLOBAL_LATENCIES[-1] - avg_latency_val
        delta_str = f"{delta_val:+.0f}ms"
    else:
        avg_latency = "N/A"
        delta_str = "0ms"
        
    col3.metric("Avg Backend Latency", avg_latency, delta_str, delta_color="inverse")
    
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
            st.markdown(f"🚨 **ACTIVE ALERTS:** {', '.join(active_alerts)} *(AI adjustments applied)*")
        else:
            st.markdown("✅ No critical alerts affecting traffic.")
            
        st.markdown("---")
        
        for n in latest_news:
            st.markdown(f"**{n.get('title', '')}**")
            st.caption(f"{n.get('source', '')} • {n.get('pub_date', '')}")
            st.markdown("")
    else:
        st.info("Awaiting live news sync...")

# Auto-refresh every 2 seconds to show new incidents
time.sleep(2)
st.rerun()
