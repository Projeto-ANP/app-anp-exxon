import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np

# Function to get latitude and longitude of a city using Geopy
def get_lat_long(location):
    geolocator = Nominatim(user_agent="my_geocoder")  # Set a valid user agent
    location = geolocator.geocode(location)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Function to get latitude and longitude of a location in Brazil
def get_lat_long_brazil(city, state):
    # Construct location string with city and state
    location = f"{city}, {state}, Brazil"
    # Get latitude and longitude
    return get_lat_long(location)

# Sample data
products = ['Product A', 'Product B', 'Product C']

num_rows = 100

np.random.seed(42)
years = np.random.randint(2010, 2025, num_rows)
months = np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], num_rows)

# Generate a number of unique states corresponding to the number of rows
states = np.random.choice(['São Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Rio Grande do Sul', 'Paraná'], num_rows)

# Dictionary mapping states to their cities
cities_by_state = {
    'São Paulo': ['São Paulo', 'Campinas', 'Santo André', 'Guarulhos', 'São Bernardo do Campo'],
    'Rio de Janeiro': ['Rio de Janeiro', 'Nova Iguaçu', 'Duque de Caxias', 'Niterói', 'São Gonçalo'],
    'Minas Gerais': ['Belo Horizonte', 'Uberlândia', 'Contagem', 'Juiz de Fora', 'Betim'],
    'Rio Grande do Sul': ['Porto Alegre', 'Caxias do Sul', 'Pelotas', 'Canoas', 'Novo Hamburgo'],
    'Paraná': ['Curitiba', 'Londrina', 'Maringá', 'Pontal do Paraná', 'Ponta Grossa']
}

# Generate data with cities corresponding to their states
cities = [np.random.choice(cities_by_state[state]) for state in states]
products_random = np.random.choice(products, num_rows)
value = np.random.randint(1, 150, num_rows)

data = pd.DataFrame({
    'Year': years,
    'Month': months,
    'State': states,
    'City': cities,
    'Product': products_random,
    'Value': value
})

# Add latitude and longitude columns to the DataFrame
data['LAT'], data['LON'] = zip(*data.apply(lambda row: get_lat_long_brazil(row['City'], row['State']), axis=1))

# Save DataFrame to Excel file
data.to_excel("test_data.xlsx", index=False)