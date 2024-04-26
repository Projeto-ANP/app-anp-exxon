import streamlit as st 
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import Search
import geobr

# Set page config
st.set_page_config(page_title='Dashboard', layout='wide')

# Read Brazil data
brasil = geobr.read_country(year=2020)

# Calculate center and zoom level for Brazil
center_lat = -14.235004  # Latitude centrada no Brasil
center_lon = -51.92528    # Longitude centrada no Brasil
zoom_level = 5           # Nível de zoom

# Set up sidebar
st.title('Highway Dashboard')

# Create the map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=zoom_level,
    control_scale=True,
)

# Add basemap
folium.TileLayer('CartoDB dark_matter').add_to(m)

# Add Brazil data
brasil_geojson = brasil.to_crs("EPSG:4326").to_json()

folium.GeoJson(
    brasil_geojson,
    name='Brasil',
    style_function=lambda x: {'color': '#7fcdbb', 'fillOpacity': 0.3, 'weight': 0.5},
    highlight_function=lambda x: {'weight':3, 'fillColor':None, 'color': 'Yellow'},
    smooth_factor=2.0,
    zoom_on_click=False,
    tooltip=folium.GeoJsonTooltip(fields=['name_state'], aliases=['Estado'], labels=True, sticky=False),
    popup = folium.GeoJsonPopup(fields=['name_state'], aliases=['Estado'], labels=True, sticky=False),
    popup_keep_highlighted=True
).add_to(m)

# Render the map on Streamlit with increased width
folium_static(m, width=1200, height=1200)
