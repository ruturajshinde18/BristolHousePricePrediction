import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime

# ----------------------------------------------------------------------------
# Config & constants
# ----------------------------------------------------------------------------

API_URL = "http://localhost:8000"

# Bristol bounding box (approx.)
BRISTOL_BOUNDS = {
    "min_lat": 51.35,
    "max_lat": 51.55,
    "min_lon": -2.75,
    "max_lon": -2.45,
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def is_in_bristol(lat: float, lon: float) -> bool:
    """Return True if (lat, lon) sits inside our Bristol bounding box."""
    return (
        BRISTOL_BOUNDS["min_lat"] <= lat <= BRISTOL_BOUNDS["max_lat"]
        and BRISTOL_BOUNDS["min_lon"] <= lon <= BRISTOL_BOUNDS["max_lon"]
    )


def call_prediction_api(payload: dict):
    """POST to the FastAPI service and return JSON or raise exception."""
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------------
# Streamlit page setup
# ----------------------------------------------------------------------------

st.set_page_config("Bristol House Price Predictor", "🏠", layout="wide")
st.title("🏠 Bristol House Price Predictor")

st.markdown(
    """Click inside the red rectangle **within Bristol** to get a house‑price
    prediction.  Manual coordinates will work only if they are inside the
    rectangle.  Clicks/pans elsewhere are ignored."""
)

# ----------------------------------------------------------------------------
# Session‑state initialisation
# ----------------------------------------------------------------------------

for key in (
    "clicked_lat",
    "clicked_lon",
    "prediction",
    "prediction_time",
):
    st.session_state.setdefault(key, None)

# ----------------------------------------------------------------------------
# Sidebar — property parameters
# ----------------------------------------------------------------------------

st.sidebar.header("Property Details")

property_types = {
    "Detached": "D",
    "Semi‑Detached": "S",
    "Terraced": "T",
    "Flat/Maisonette": "F",
    "Other": "O",
}
prop_disp = st.sidebar.selectbox("Property Type", list(property_types))
prop_code = property_types[prop_disp]

new_build_disp = st.sidebar.selectbox("New Build", ["No", "Yes"])
new_build_code = "Y" if new_build_disp == "Yes" else "N"

tenure_types = {"Freehold": "F", "Leasehold": "L"}
tenure_disp = st.sidebar.selectbox("Tenure", list(tenure_types))
tenure_code = tenure_types[tenure_disp]
now = datetime.now()
year = st.sidebar.number_input("Year", 2000, 2025, now.year)

# ---------------------------------------------------------------------------
# Map setup (Bristol‑only)
# ---------------------------------------------------------------------------

bristol_centre = [51.4545, -2.5879]

m = folium.Map(
    location=bristol_centre,
    zoom_start=12,
    tiles="OpenStreetMap",
    max_bounds=True,  # stop user from panning far away
)

# Draw a rectangle boundary to visualise bounds
folium.Rectangle(
    [[BRISTOL_BOUNDS["min_lat"], BRISTOL_BOUNDS["min_lon"]],
     [BRISTOL_BOUNDS["max_lat"], BRISTOL_BOUNDS["max_lon"]]],
    color="red",
    fill=False,
    weight=2,
).add_to(m)

# Add a marker for the city centre
folium.Marker(
    bristol_centre,
    tooltip="Bristol City Centre",
    icon=folium.Icon(color="red", icon="star"),
).add_to(m)

# Show map & capture clicks
map_data = st_folium(
    m,
    width=1200,
    height=500,
    returned_objects=["last_clicked"],
    key="map1",
)

# ---------------------------------------------------------------------------
# Handle map clicks — only accept Bristol ones
# ---------------------------------------------------------------------------

if map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    if is_in_bristol(lat, lon):
        st.session_state.clicked_lat = lat
        st.session_state.clicked_lon = lon
    else:
        st.warning("Click ignored — please choose a spot inside Bristol bounds.")

# ---------------------------------------------------------------------------
# Manual coordinate entry
# ---------------------------------------------------------------------------

st.subheader("📍Location Selection ")
col1, col2 = st.columns(2)

with col1:
    manual_lat = st.number_input("Latitude", format="%.6f", value=
        st.session_state.clicked_lat or bristol_centre[0])
with col2:
    manual_lon = st.number_input("Longitude", format="%.6f", value=
        st.session_state.clicked_lon or bristol_centre[1])

# Use manual coordinates if valid and changed
if (
    (manual_lat, manual_lon) != (st.session_state.clicked_lat, st.session_state.clicked_lon)
    and is_in_bristol(manual_lat, manual_lon)
):
    st.session_state.clicked_lat = manual_lat
    st.session_state.clicked_lon = manual_lon
elif not is_in_bristol(manual_lat, manual_lon):
    st.info("Manual coordinates must be within Bristol bounds.")

# ---------------------------------------------------------------------------
# Show selected point (if any)
# ---------------------------------------------------------------------------

valid_location = (
    st.session_state.clicked_lat is not None and
    st.session_state.clicked_lon is not None and
    is_in_bristol(st.session_state.clicked_lat, st.session_state.clicked_lon)
)

if valid_location:
    st.success(f"📍 Location: {st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lon:.6f}")
else:
    st.info("Choose a location inside the Bristol rectangle above.")

# ---------------------------------------------------------------------------
# Predict button — disabled unless we have a valid location
# ---------------------------------------------------------------------------

def do_predict():
    payload = {
        "latitude": st.session_state.clicked_lat,
        "longitude": st.session_state.clicked_lon,
        "property_type": prop_code,
        "new_build": new_build_code,
        "tenure": tenure_code,
        "year": int(year)
    }
    try:
        st.session_state.prediction = call_prediction_api(payload)
        st.session_state.prediction_time = datetime.now()
        st.toast("Prediction complete!", icon="🎉")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.button(
    "🔮 Predict House Price",
    type="primary",
    disabled=not valid_location,
    on_click=do_predict,
)

# ---------------------------------------------------------------------------
# Display prediction (if any)
# ---------------------------------------------------------------------------

if st.session_state.prediction:
    result = st.session_state.prediction

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", result["formatted_price"])

    with col2:
        st.info(
            f"""
            **Lat/Lon**: {st.session_state.clicked_lat:.4f}, {st.session_state.clicked_lon:.4f}  
            **Property**: {prop_disp}  
            **Tenure**: {tenure_disp}  
            **New Build**: {new_build_disp}  
            **Date**: {year}
            """
        )

    # Mini map with marker
    mini = folium.Map(location=[st.session_state.clicked_lat, st.session_state.clicked_lon], zoom_start=15)
    folium.Marker(
        [st.session_state.clicked_lat, st.session_state.clicked_lon],
        tooltip=result["formatted_price"],
    ).add_to(mini)
    st_folium(mini, width=700, height=350, key="mini_map")

    if st.button("🗑️ Clear Prediction", type="secondary"):
        st.session_state.prediction = None
        st.session_state.prediction_time = None
        st.rerun()

# ---------------------------------------------------------------------------
# Quick‑select buttons for 4 popular neighbourhoods (inside bounds)
# ---------------------------------------------------------------------------

st.subheader("🏘️ Quick Locations")
locs = {
    "Clifton": (51.4641, -2.6103),
    "Redland": (51.4711, -2.6037),
    "Southville": (51.4398, -2.6205),
    "Bedminster": (51.4325, -2.5889),
}

cols = st.columns(len(locs))
for (name, (lat, lon)), c in zip(locs.items(), cols):
    with c:
        if st.button(name):
            st.session_state.clicked_lat = lat
            st.session_state.clicked_lon = lon
            st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

