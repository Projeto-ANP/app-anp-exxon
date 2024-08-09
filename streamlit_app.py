import warnings
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Filter out the specific warnings
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information*")
warnings.filterwarnings("ignore", message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.*")
warnings.filterwarnings("ignore", message="No supported index is available.*")

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path, sep=";")

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

def display_data_overview(data):
    """Display data overview."""
    st.header("Base de Dados")
    data

def display_plot():
    """Display plot."""
    st.header("Dashboard")

def decompose_time_series(data):
    """Decompose time series."""
    st.title('Decomposição de Séries Temporais')
    st.markdown("""
        Para gerar a decomposição de uma série temporal, selecione apenas 1 estado e um produto específico.
    """)

    example_data = data[['DATA', 'ESTADO', 'PRODUTO', 'QUANTIDADE_M3']]

    # Inicialize o session state se ainda não estiver configurado
    if 'selected_state' not in st.session_state:
        st.session_state['selected_state'] = example_data['ESTADO'].unique()[0]
    if 'selected_product' not in st.session_state:
        st.session_state['selected_product'] = example_data['PRODUTO'].unique()[0]
    if 'period_date' not in st.session_state:
        st.session_state['period_date'] = (example_data['DATA'].min(), example_data['DATA'].max())
    if 'button_disabled' not in st.session_state:
        st.session_state['button_disabled'] = False
    if 'explanation_text' not in st.session_state:
        st.session_state['explanation_text'] = ""

    selected_state = st.selectbox(
        'Select a state:', example_data['ESTADO'].unique(), key='selected_state')
    selected_product = st.selectbox(
        'Select a product:', example_data['PRODUTO'].unique(), key='selected_product')
    period_date = st.date_input("Pick a date", value=st.session_state['period_date'], 
                                min_value=example_data['DATA'].min(), 
                                max_value=example_data['DATA'].max(), 
                                format="MM.DD.YYYY", key='period_date')

    if st.button('Gerar'):
        # Validation and generation
        if validate_fields(selected_state, selected_product, period_date):
            period_date_start = pd.to_datetime(period_date[0])
            period_date_end = pd.to_datetime(period_date[1])

            # Filtra os dados com base nas seleções
            filtered_data = example_data[
                (example_data['ESTADO'] == selected_state) & 
                (example_data['PRODUTO'] == selected_product) &
                (example_data['DATA'] >= period_date_start) &
                (example_data['DATA'] <= period_date_end)]
    

            # Plot da série temporal original
            fig_original = px.line(filtered_data, x=filtered_data["DATA"],
                                y='QUANTIDADE_M3', title='Original Time Series')

            # Salva o gráfico na sessão
            st.session_state['fig_original'] = fig_original

            # Prepara os dados para análise de séries temporais
            example_ts = filtered_data.set_index('DATA')['QUANTIDADE_M3']

            # Decomposição da série temporal
            decomposition = seasonal_decompose(example_ts, model='additive', period=12)

            # Plot dos componentes da decomposição
            fig_trend = px.line(x=decomposition.trend.index, y=decomposition.trend)
            fig_seasonal = px.line(x=decomposition.seasonal.index, y=decomposition.seasonal)
            fig_residual = px.scatter(x=decomposition.resid.index, y=decomposition.resid)

            # Estilo dos gráficos
            fig_trend.update_layout(title=f'Trend Component of product {selected_product} in state {selected_state}',
                                    xaxis_title='Date',
                                    yaxis_title='Quantity (cubic meters)')
            fig_seasonal.update_layout(title=f'Seasonal Component of product {selected_product} in state {selected_state}',
                                    xaxis_title='Date',
                                    yaxis_title='Quantity (cubic meters)')
            fig_residual.update_layout(title=f'Residual Component {selected_product} in state {selected_state}',
                                    xaxis_title='Date',
                                    yaxis_title='Quantity (cubic meters)')

            # Salva os gráficos na sessão
            st.session_state['fig_trend'] = fig_trend
            st.session_state['fig_seasonal'] = fig_seasonal
            st.session_state['fig_residual'] = fig_residual

            # Define o estado para permitir a geração das explicações
            st.session_state['generated'] = True

    # Se os gráficos já foram gerados, exiba-os
    if st.session_state.get('generated', False):
        st.plotly_chart(st.session_state['fig_original'])
        st.plotly_chart(st.session_state['fig_trend'])
        st.plotly_chart(st.session_state['fig_seasonal'])
        st.plotly_chart(st.session_state['fig_residual'])
        
    # Mostra a explicação gerada
    if st.session_state['explanation_text']:
        st.markdown(st.session_state['explanation_text'])

def gerar_explicacoes(period_date_start, period_date_end):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f'Por que o céu é azul?')
    return response.text  # Retorna o texto gerado


def main():
    """Main function."""
    data = load_data("Database//combined_data.csv")
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    genai.configure(api_key=API_KEY)

    data.rename(columns={
        'timestamp': 'DATA', 
        'state': 'ESTADO', 
        'product': 'PRODUTO', 
        'm3': 'QUANTIDADE_M3'
    }, inplace=True)
    
    # Convert the date column to datetime format
    data['DATA'] = pd.to_datetime(data['DATA'])
    
    st.set_page_config(layout="wide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Início", "Dashboard", "Decomposição de Séries Temporais", "Base de Dados"])
    
    with tab1:
        st.header("Projeto")
        st.markdown("""
            Este site foi desenvolvido com o objetivo de explorar e entender de forma mais clara as séries temporais relacionadas à quantidade em metros cúbicos (m³) de diferentes derivados nos estados brasileiros. A plataforma é uma ferramenta interna, destinada exclusivamente aos pesquisadores vinculados ao CISIA - PUCPR (Centro Integrado de Soluções em Inteligência Artificial). Ela possibilita uma análise detalhada e precisa dos dados, facilitando a tomada de decisões informadas em estudos e pesquisas científicas.
        """)

    with tab2:
        display_plot()

    with tab3:
        decompose_time_series(data)
        
    with tab4:
        display_data_overview(data)

if __name__ == "__main__":
    main()