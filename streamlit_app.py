import warnings
import pandas as pd
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

# Filter out the specific warnings
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information*")
warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.*")
warnings.filterwarnings("ignore", message="No supported index is available.*")


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path, sep=",")

def is_valid_date(date_str):
    """Function to check if the input is a valid date"""
    try:
        pd.to_datetime(date_str)
        return True
    except ValueError:
        return False

def validate_fields(state, product, period_date):
    """Check if all fields are filled and if the dates are in the correct position"""
    if state and product and len(period_date) == 2:
        if not is_valid_date(period_date[0]):
            st.warning("Invalid start date!")
            return False
        if not is_valid_date(period_date[1]):
            st.warning("Invalid end date!")
            return False
        
        # Convert dates to datetime objects
        start_date = pd.to_datetime(period_date[0])
        end_date = pd.to_datetime(period_date[1])

        # Check if the period is longer than 24 months
        if (end_date - start_date).days <= 24 * 30:  # assuming 30 days per month
            st.warning("The period cannot be less than 24 months! This is because the time series used for seasonal decomposition doesn't have enough data for accurate seasonal decomposition. The 'season_decompose' method from the Statsmodels library requires at least 24 observations to calculate seasonal decomposition accurately.")
            return False

        return True
    else:
        st.warning("Invalid date!")
        return False
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
    st.header("Dashboard")

    # Box plot comparing resale prices between regions
    regional_fig = px.bar(dynamic_filters.filter_df(), x="ESTADO", y="QUANTIDADE0M3",
                          color="REGIAO", title="Quantity by State")
    st.plotly_chart(regional_fig)

    # Plot distribution of average resale prices by state
    fig_state_price = px.box(dynamic_filters.filter_df(), x='ESTADO', y='PRECO MEDIO REVENDA',
                             color="REGIAO", title='Distribution of Average Resale Prices by State')
    st.plotly_chart(fig_state_price)

    # Correlation Analysis
    st.header("Correlation Heatmap")

    filtered_df = dynamic_filters.filter_df()
    numeric_columns = filtered_df.select_dtypes(
        include=['number']).columns.tolist()

    if len(numeric_columns) >= 2:  # Ensure there are at least two numeric columns for correlation
        correlation_matrix = filtered_df[numeric_columns].corr()
        correlation_fig = px.imshow(correlation_matrix)
        # Adjusting the size of the correlation heatmap
        correlation_fig.update_layout(width=1000, height=800)
        st.plotly_chart(correlation_fig)

    else:
        st.warning("Insufficient numeric data to compute correlation.")


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

    min_date = example_data['DATA'].min()
    max_date = example_data['DATA'].max()
    period_date = st.date_input("Pick a date", (min_date, max_date), min_value=min_date, max_value=max_date, format="MM.DD.YYYY")
        
    if st.button('Generate'):
        # Validation and generation
        if validate_fields(selected_state, selected_product, period_date):
            
            # Assuming period_date[0] and period_date[1] are in date format
            period_date_start = pd.to_datetime(period_date[0])
            period_date_end = pd.to_datetime(period_date[1])

            # Filter the data according to selections
            filtered_data = example_data[
                (example_data['ESTADO'] == selected_state) & 
                (example_data['PRODUTO'] == selected_product) &
                (example_data['DATA'] >= period_date_start) &
                (example_data['DATA'] <= period_date_end)]

            # Plot the original time series
            fig_original = px.line(filtered_data, x=filtered_data["DATA"],
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
            # sarima_forecast(example_ts, selected_product, selected_state)
        else:
            st.warning("There's an issue with the filters, please check and try again.")
        
def main():
    """Main function."""
    data = load_data("Database//UF-072001-022024.csv")
    data = apply_month_mappings(data)
    data.drop(['DIA'], axis=1, inplace=True)
    
    # Convert the date column to datetime format
    data['DATA'] = pd.to_datetime(data['DATA'])
    
    st.set_page_config(layout="wide")

    with st.sidebar:
        display_filters_sidebar(data)

    dynamic_filters = DynamicFilters(
        df=data, filters=['ANO', 'MES', 'REGIAO', 'ESTADO', 'PRODUTO'])
    
    tab1, tab2, tab3, tab4 = st.tabs(["Home", "Dashboard", "Time Series Decomposition", "Data Overview"])
    
    with tab1:
        st.header("Project")
        st.markdown("""
            This dashboard provides information about the sales of oil and its derivatives. 
            Explore different filters and visualize data dynamically.
        """)

    with tab2:
        display_plot(dynamic_filters)

    with tab3:
        decompose_time_series(data)
        
    with tab4:
        display_data_overview(dynamic_filters)

if __name__ == "__main__":
    main()