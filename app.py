
import math
from pathlib import Path

import requests
import streamlit as st

# ----------------------------------------
# APP CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Agriculture System")
st.caption("Crop guidance • Soil monitoring • Weather • AI detection")

# Optional AI models. Add these files if available:
# models/snake.pt
# models/weed.pt
MODEL_DIR = Path("models")


# ----------------------------------------
# CROP RECOMMENDATION
# ----------------------------------------
CROPS = {
    "Black": ["Cotton", "Soybean", "Sorghum"],
    "Red": ["Groundnut", "Millets", "Pulses"],
    "Sandy": ["Groundnut", "Watermelon", "Millets"],
    "Alluvial": ["Rice", "Wheat", "Sugarcane"],
}

FERTILIZERS = {
    "Rice": "Nitrogen, phosphorus and potassium as indicated by soil testing.",
    "Wheat": "Nitrogen and phosphorus as indicated by soil testing.",
    "Cotton": "Balanced NPK based on soil test and crop growth stage.",
    "Maize": "Nitrogen and phosphorus based on soil test.",
    "Tomato": "Balanced NPK and potassium based on soil test.",
    "Soybean": "Use a locally recommended nutrient plan based on soil testing.",
    "Groundnut": "Use crop-specific nutrient and calcium guidance from local experts.",
    "Sorghum": "Nitrogen and phosphorus according to local soil-test recommendations.",
    "Millets": "Use a locally suitable nutrient plan based on soil testing.",
    "Pulses": "Use crop-specific recommendations; avoid unnecessary nitrogen.",
    "Watermelon": "Balanced crop nutrition based on soil test and growth stage.",
    "Sugarcane": "Split nutrient applications according to local crop guidance.",
}


# ----------------------------------------
# SOIL MOISTURE
# ----------------------------------------
def show_moisture(moisture):
    st.metric("Soil moisture", f"{moisture:.1f}%")
    st.progress(int(max(0, min(100, moisture))))

    # These are illustrative thresholds, not universal crop settings.
    if moisture < 30:
        st.warning(
            "Low moisture reading. Check the sensor and crop water needs; "
            "irrigation may be required."
        )
    elif moisture < 70:
        st.info(
            "Moderate moisture reading. Compare with the needs of your crop."
        )
    else:
        st.success(
            "High moisture reading. Check drainage and avoid overwatering."
        )


def get_esp32_moisture(url):
    """Expected endpoint response: {"moisture_percent": 42.5}"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    value = float(response.json()["moisture_percent"])

    if not 0 <= value <= 100:
        raise ValueError("Sensor percentage must be between 0 and 100.")

    return value


# ----------------------------------------
# RAIN CHECK WITHIN 20 KM
# ----------------------------------------
def distance_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    a = max(0.0, min(1.0, a))

    return 2 * earth_radius * math.asin(math.sqrt(a))


def check_nearby_rain(center_lat, center_lon):
    # Sample the center and nearby points on a 5 km grid.
    # This checks selected points, not every location in the radius.
    step_lat = 5.0 / 111.0
    step_lon = 5.0 / (
        111.0 * max(0.1, abs(math.cos(math.radians(center_lat))))
    )

    points = [(center_lat, center_lon)]

    for i in range(-4, 5):
        for j in range(-4, 5):
            lat = center_lat + i * step_lat
            lon = center_lon + j * step_lon

            if distance_km(
                center_lat, center_lon, lat, lon
            ) <= 20:
                points.append((lat, lon))

    # Limit API calls and keep the points spread over the area.
    # A denser grid can be used with caching and API-rate awareness.
    points = list(dict.fromkeys(points))

    results = []
    url = "https://api.open-meteo.com/v1/forecast"

    for lat, lon in points:
        response = requests.get(
            url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "rain,precipitation",
                "timezone": "auto",
            },
            timeout=12,
        )
        response.raise_for_status()
        current = response.json()["current"]

        rain = float(current.get("rain") or 0)
        precipitation = float(current.get("precipitation") or 0)

        if rain > 0 or precipitation > 0:
            results.append({
                "distance_km": round(
                    distance_km(center_lat, center_lon, lat, lon), 1
                ),
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "rain_mm": rain,
                "precipitation_mm": precipitation,
            })

    return results, len(points)


# ----------------------------------------
# AI IMAGE DETECTION
# ----------------------------------------
def detect_image(upload, model_path, title):
    if not model_path.exists():
        st.error(
            f"Model not found: {model_path}. "
            "Add a compatible trained model to the models folder."
        )
        return

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model(upload.getvalue())

        result = results[0]
        st.image(
            result.plot(),
            channels="BGR",
            caption=f"{title} result"
        )

        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                st.write(
                    f"{names[class_id]} — confidence {confidence:.0%}"
                )
        else:
            st.info("No objects detected by this model.")

        st.caption(
            "AI results can be wrong. Do not approach a suspected snake "
            "based on this result; keep away and contact trained personnel."
        )

    except Exception as error:
        st.error(f"Could not run the model: {error}")


# ----------------------------------------
# APP NAVIGATION
# ----------------------------------------
feature = st.sidebar.radio(
    "Choose a feature",
    [
        "Home",
        "Crop recommendation",
        "Soil moisture",
        "Fertilizer guide",
        "Rain within 20 km",
        "Snake detection",
        "Weed detection",
    ],
)


if feature == "Home":
    st.subheader("Your farming dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Features", "6")
    c2.metric("Rain radius", "20 km")
    c3.metric("Soil readings", "Manual + sensor")

    st.write(
        "Select a feature from the sidebar to begin. "
        "Sensor and AI results require compatible hardware or models."
    )


elif feature == "Crop recommendation":
    st.subheader("🌾 Crop recommendation")

    soil = st.selectbox("Choose soil type", list(CROPS.keys()))

    st.write("Example crops for this soil type:")
    for crop in CROPS[soil]:
        st.write(f"- {crop}")

    st.caption(
        "These are general examples. Confirm suitability using soil pH, "
        "nutrients, rainfall, irrigation, season and local advice."
    )


elif feature == "Soil moisture":
    st.subheader("💧 Soil moisture monitoring")

    mode = st.radio(
        "Reading method",
        ["Manual entry", "ESP32 sensor URL"]
    )

    if mode == "Manual entry":
        moisture = st.slider(
            "Soil moisture (%)",
            min_value=0,
            max_value=100,
            value=40
        )
        show_moisture(moisture)

    else:
        sensor_url = st.text_input(
            "ESP32 endpoint URL",
            placeholder="http://192.168.1.50/moisture"
        )

        st.caption(
            'The endpoint must return JSON such as '
            '{"moisture_percent": 42.5}.'
        )

        if st.button("Read sensor"):
            if not sensor_url.startswith(("http://", "https://")):
                st.error("Enter a valid sensor URL.")
            else:
                try:
                    moisture = get_esp32_moisture(sensor_url)
                    show_moisture(moisture)
                except Exception as error:
                    st.error(f"Could not read sensor: {error}")


elif feature == "Fertilizer guide":
    st.subheader("🧪 Fertilizer guide")

    crop = st.selectbox("Choose crop", list(FERTILIZERS.keys()))
    st.write("**General guidance:**")
    st.write(FERTILIZERS[crop])

    st.warning(
        "This app does not calculate a fertilizer dose. "
        "Use a soil test and your local agricultural recommendation "
        "before applying fertilizer."
    )


elif feature == "Rain within 20 km":
    st.subheader("🌧️ Nearby rain check")

    lat = st.number_input(
        "Center latitude",
        min_value=-90.0,
        max_value=90.0,
        value=17.3850,
        format="%.5f"
    )
    lon = st.number_input(
        "Center longitude",
        min_value=-180.0,
        max_value=180.0,
        value=78.4867,
        format="%.5f"
    )

    if st.button("Check nearby weather"):
        with st.spinner("Checking weather at nearby points..."):
            try:
                rain_points, checked = check_nearby_rain(lat, lon)

                st.write(f"Checked {checked} sampled locations.")

                if rain_points:
                    st.warning("Rain is reported at these sampled locations.")
                    st.dataframe(rain_points, use_container_width=True)
                else:
                    st.info(
                        "No rain reported at the sampled locations. "
                        "This does not guarantee that the whole area is dry."
                    )

            except Exception as error:
                st.error(f"Weather request failed: {error}")

    st.caption(
        "Weather-model data is not a local rain sensor or a radar image. "
        "It may miss small, localized showers."
    )


elif feature == "Snake detection":
    st.subheader("🐍 Snake image detection")

    upload = st.file_uploader(
        "Upload a snake or outdoor photo",
        type=["jpg", "jpeg", "png"],
        key="snake_image"
    )

    if upload:
        st.image(upload, caption="Input image", use_container_width=True)

        if st.button("Analyze image"):
            detect_image(
                upload,
                MODEL_DIR / "snake.pt",
                "Snake detection"
            )


elif feature == "Weed detection":
    st.subheader("🌿 Crop and weed image detection")

    upload = st.file_uploader(
        "Upload a crop-field photo",
        type=["jpg", "jpeg", "png"],
        key="weed_image"
    )

    if upload:
        st.image(upload, caption="Input image", use_container_width=True)

        if st.button("Analyze field"):
            detect_image(
                upload,
                MODEL_DIR / "weed.pt",
                "Weed detection"
            )
            
