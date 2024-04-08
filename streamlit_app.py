import pandas as pd
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path, sep=",")


def apply_month_mappings(data):
    """Apply month names mapping."""
    months = {
        1: 'JANUARY', 2: 'FEBRUARY', 3: 'MARCH', 4: 'APRIL',
        5: 'MAY', 6: 'JUNE', 7: 'JULY', 8: 'AUGUST',
        9: 'SEPTEMBER', 10: 'OCTOBER', 11: 'NOVEMBER', 12: 'DECEMBER'
    }
    data['MES'] = data['MES'].map(months)
    return data


def display_filters_sidebar(data):
    """Display sidebar filters."""
    st.title("Filters")
    dynamic_filters = DynamicFilters(
        df=data, filters=['ANO', 'MES', 'REGIAO', 'ESTADO', 'PRODUTO'])
    dynamic_filters.display_filters(location="sidebar", gap="large")
    if st.button("Reset All filters"):
        dynamic_filters.reset_filters()


def display_data_overview(dynamic_filters):
    """Display data overview."""
    st.header("Data Overview")
    dynamic_filters.display_df()


def display_plot(dynamic_filters):
    """Display plot."""

    # Regional Analysis
    st.header("Regional Analysis")

    # Box plot comparing resale prices between regions
    regional_fig = px.box(dynamic_filters.filter_df(), x="REGIAO", y="PRECO MEDIO REVENDA", color="REGIAO", title="Resale Price Distribution by Region")
    st.plotly_chart(regional_fig)

    # Plot distribution of average resale prices by state
    fig_state_price = px.box(dynamic_filters.filter_df(), x='ESTADO', y='PRECO MEDIO REVENDA', color="REGIAO", title='Distribution of Average Resale Prices by State')
    st.plotly_chart(fig_state_price)

    # Correlation Analysis
    st.header("Correlation Analysis")

    # Correlation heatmap between different variables
    correlation_matrix = dynamic_filters.filter_df().corr()
    correlation_fig = px.imshow(correlation_matrix, title="Correlation Heatmap")
    correlation_fig.update_layout(width=1000, height=800)  # Adjusting the size of the correlation heatmap
    st.plotly_chart(correlation_fig)
    
    # Clear layout settings after plotting graph 1
    correlation_fig.update_layout(title=None, xaxis_title=None, yaxis_title=None, width=None, height=None)



def sarima_forecast(example_ts, selected_product, selected_state):
    """ ======= SAMIRA =======

    # The seasonal_decompose already suggested a yearly seasonality (period=12 months), 
    # so let's use that as a starting point for the SARIMA model.
    # We'll start with the same (1,1,1) configuration for ARIMA parameters and add a simple seasonal component.

    """

    # Define SARIMA model configuration
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    # Fit the SARIMA model
    sarima_model = SARIMAX(example_ts, order=sarima_order,
                           seasonal_order=seasonal_order)
    sarima_model_fit = sarima_model.fit(disp=False)

    # Forecast the next 12 months with the SARIMA model
    sarima_forecast = sarima_model_fit.forecast(steps=12)

    # To ensure the forecast aligns correctly on the timeline, we create a new date range starting from the last date
    # of the historical data
    forecast_dates = pd.date_range(
        start=example_ts.index[-1], periods=len(sarima_forecast) + 1, freq='M')[1:]

    # Plotting the corrected visualization using Plotly Express
    fig = px.line()
    fig.add_scatter(x=example_ts.index, y=example_ts,
                    mode='lines+markers', name='Original')
    fig.add_scatter(x=forecast_dates, y=sarima_forecast, mode='lines+markers',
                    name='SARIMA Forecast', line=dict(color='red'))
    fig.update_layout(title=f'Corrected Forecast for {selected_product} in {selected_state} using SARIMA Model',
                      xaxis_title='Date',
                      yaxis_title='Quantity (cubic meters)',
                      width=1000, height=500
                      )

    st.plotly_chart(fig)



def decompose_time_series(data):
    """Decompose time series."""
    st.title('Time Series Decomposition')
    st.markdown("""
        To generate the decomposition of a time series, select only 1 state and one specific product.
    """)

    example_data = data[['DATA', 'ESTADO', 'PRODUTO', 'QUANTIDADE0M3']]

    selected_state = st.selectbox(
        'Select a state:', example_data['ESTADO'].unique())
    selected_product = st.selectbox(
        'Select a product:', example_data['PRODUTO'].unique())

    if st.button('Generate'):
        # Filter the data according to selections
        filtered_data = example_data[(example_data['ESTADO'] == selected_state) & (
            example_data['PRODUTO'] == selected_product)]

        # Plot the original time series
        fig_original = px.line(filtered_data, x=filtered_data.index,
                               y='QUANTIDADE0M3', title='Original Time Series')
        st.plotly_chart(fig_original)

        # Prepare data for time series analysis
        example_ts = filtered_data.set_index('DATA')['QUANTIDADE0M3']

        # Decompose the time series
        decomposition = seasonal_decompose(
            example_ts, model='additive', period=12)

        # Plot the decomposition components
        fig_trend = px.line(x=decomposition.trend.index, y=decomposition.trend)
        fig_seasonal = px.line(
            x=decomposition.seasonal.index, y=decomposition.seasonal)
        fig_residual = px.scatter(
            x=decomposition.resid.index, y=decomposition.resid)

        # defining plot style
        fig_trend.update_layout(title=f'Trend Component of product {selected_product} in state {selected_state}',
                                xaxis_title='Date',
                                yaxis_title='Quantity (cubic meters)')
        fig_seasonal.update_layout(title=f'Seasonal Component of product {selected_product} in state {selected_state}',
                                   xaxis_title='Date',
                                   yaxis_title='Quantity (cubic meters)')
        fig_residual.update_layout(title=f'Residual Component {selected_product} in state {selected_state}',
                                   xaxis_title='Date',
                                   yaxis_title='Quantity (cubic meters)')

        # Display the component graphs
        st.subheader('Decomposed Time Series Components')
        st.plotly_chart(fig_trend)
        st.plotly_chart(fig_seasonal)
        st.plotly_chart(fig_residual)

        # Plot SAMIRA
        sarima_forecast(example_ts, selected_product, selected_state)


def main():
    """Main function."""
    
    st.title("ANP and ExxonMobil Dashboard")
    st.markdown("""
        This dashboard provides insights into ANP and ExxonMobil data.
        Explore different filters and visualize the data dynamically.
    """)

    data = load_data("Database//UF-072001-022024.csv")
    data = apply_month_mappings(data)
    data.drop(['DIA'], axis=1, inplace=True)
    
    with st.sidebar:
        display_filters_sidebar(data)

    dynamic_filters = DynamicFilters(
        df=data, filters=['ANO', 'MES', 'REGIAO', 'ESTADO', 'PRODUTO'])

    display_data_overview(dynamic_filters)
    display_plot(dynamic_filters)
    decompose_time_series(data)


if __name__ == "__main__":
    main()