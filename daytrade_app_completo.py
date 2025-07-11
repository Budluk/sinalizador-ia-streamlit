# -*- coding: utf-8 -*-
"""
APLICAÇÃO STREAMLIT - SINALIZADOR DE DAYTRADE COM IA (VERSÃO CORRIGIDA)

Esta aplicação cria uma interface web para visualizar os sinais de compra/venda
gerados por um modelo de Machine Learning treinado.

Alteração:
- Adicionado tld='com' na conexão com a Binance para evitar erros de restrição geográfica.
"""

# --- Importações ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import datetime
from binance.client import Client as BinanceClient
import ta

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Sinalizador de IA para Daytrade",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES ---
FEATURES_LIST = [
    'retorno', 'media9', 'media21', 'volume', 'volume_ma', 'volume_diff_ma', 'volume_ratio_ma', 'volume_anomalo',
    'range_vela', 'close_pos_pct', 'open_pos_pct', 'body_size', 'body_ratio_range',
    'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio',
    'pressao_compra', 'pressao_venda',
    'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 'adx',
    'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbw', 'bb_bbp', 'obv', 'mfi'
]
MODELO_PATH = 'modelo_ia.pkl'
METADATA_PATH = 'metadata_modelo.json'

# --- FUNÇÕES AUXILIARES ---

@st.cache_resource
def carregar_modelo_e_metadata():
    """Carrega o modelo e os metadados uma única vez para otimizar a performance."""
    try:
        modelo = joblib.load(MODELO_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return modelo, metadata
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo ('{MODELO_PATH}' ou '{METADATA_PATH}') não encontrado.")
        st.info("Certifique-se de que os arquivos do modelo treinado (`modelo_ia.pkl` e `metadata_modelo.json`) foram enviados para o GitHub.")
        return None, None

@st.cache_resource
def conectar_cliente_binance():
    """Conecta-se à API da Binance usando as chaves armazenadas no Streamlit Secrets."""
    try:
        api_key = st.secrets["binance"]["BINANCE_API_KEY"]
        api_secret = st.secrets["binance"]["BINANCE_API_SECRET"]
        # **CORREÇÃO APLICADA AQUI**
        # Adicionado tld='com' para especificar o domínio global da API e evitar bloqueios geográficos.
        client = BinanceClient(api_key, api_secret, tld='com')
        client.get_account_status()
        return client
    except Exception as e:
        st.error(f"Falha ao conectar à Binance. Verifique as suas credenciais em 'Secrets'. Erro: {e}")
        return None

def buscar_dados_recentes(_client, simbolo, intervalo, limite=100):
    """Busca os dados mais recentes de um ativo na Binance."""
    if not _client:
        return pd.DataFrame()
    try:
        klines = _client.get_klines(symbol=simbolo, interval=intervalo, limit=limite)
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['open_time'] = pd.to_datetime(df[col], unit='ms')
        df.set_index('open_time', inplace=True)
        return df
    except Exception as e:
        st.warning(f"Não foi possível buscar os dados mais recentes. Erro: {e}")
        return pd.DataFrame()

# A função de gerar features deve ser IDÊNTICA à do script de treino.
def gerar_features(df):
    """Gera features de análise técnica a partir de dados OHLCV."""
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df_copy = df.copy()
    epsilon = 1e-9

    df_copy['retorno'] = df_copy['close'].pct_change()
    df_copy['media9'] = ta.trend.sma_indicator(df_copy['close'], window=9)
    df_copy['media21'] = ta.trend.sma_indicator(df_copy['close'], window=21)
    df_copy['volume_ma'] = ta.trend.sma_indicator(df_copy['volume'], window=10)
    df_copy['volume_diff_ma'] = df_copy['volume'] - df_copy['volume_ma']
    df_copy['volume_ratio_ma'] = df_copy['volume'] / (df_copy['volume_ma'] + epsilon)
    df_copy['volume_anomalo'] = (df_copy['volume_ratio_ma'] > 1.5).astype(int)
    df_copy['range_vela'] = df_copy['high'] - df_copy['low']
    df_copy['close_pos_pct'] = (df_copy['close'] - df_copy['low']) / (df_copy['range_vela'] + epsilon)
    df_copy['open_pos_pct'] = (df_copy['open'] - df_copy['low']) / (df_copy['range_vela'] + epsilon)
    df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
    df_copy['body_ratio_range'] = df_copy['body_size'] / (df_copy['range_vela'] + epsilon)
    df_copy['upper_wick'] = df_copy['high'] - df_copy[['open', 'close']].max(axis=1)
    df_copy['lower_wick'] = df_copy[['open', 'close']].min(axis=1) - df_copy['low']
    df_copy['upper_wick_ratio'] = df_copy['upper_wick'] / (df_copy['range_vela'] + epsilon)
    df_copy['lower_wick_ratio'] = df_copy['lower_wick'] / (df_copy['range_vela'] + epsilon)
    df_copy['pressao_compra'] = (df_copy['close'] > df_copy['open']).astype(int)
    df_copy['pressao_venda'] = (df_copy['close'] < df_copy['open']).astype(int)
    df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
    stoch = ta.momentum.StochasticOscillator(df_copy['high'], df_copy['low'], df_copy['close'], window=14, smooth_window=3)
    df_copy['stoch_k'] = stoch.stoch()
    df_copy['stoch_d'] = stoch.stoch_signal()
    macd = ta.trend.MACD(df_copy['close'])
    df_copy['macd'] = macd.macd()
    df_copy['macd_signal'] = macd.macd_signal()
    df_copy['macd_diff'] = macd.macd_diff()
    df_copy['adx'] = ta.trend.adx(df_copy['high'], df_copy['low'], df_copy['close'], window=14)
    bollinger = ta.volatility.BollingerBands(df_copy['close'])
    df_copy['bb_bbm'] = bollinger.bollinger_mavg()
    df_copy['bb_bbh'] = bollinger.bollinger_hband()
    df_copy['bb_bbl'] = bollinger.bollinger_lband()
    df_copy['bb_bbw'] = bollinger.bollinger_wband()
    df_copy['bb_bbp'] = bollinger.bollinger_pband()
    df_copy['obv'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
    df_copy['mfi'] = ta.volume.money_flow_index(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14)
    
    return df_copy.dropna()

# --- INTERFACE PRINCIPAL ---

def main():
    modelo, metadata = carregar_modelo_e_metadata()
    client = conectar_cliente_binance()

    if modelo is None or metadata is None or client is None:
        st.warning("A aplicação não pode iniciar devido a erros de carregamento. Verifique as mensagens acima.")
        return

    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1621417488214-2a6046103b51?q=80&w=2832&auto=format&fit=crop", use_column_width=True)
        st.title("Informações do Modelo")
        st.info(f"**Ativo Treinado:** `{metadata.get('ativo', 'N/A')}`")
        st.info(f"**Intervalo:** `{metadata.get('intervalo', 'N/A')}`")
        st.info(f"**Data do Treino:** `{metadata.get('data_treino', 'N/A')}`")
        
        acuracia = metadata.get('accuracy_teste', 0) * 100
        st.metric(label="Acurácia do Modelo (em teste)", value=f"{acuracia:.2f}%")
        
        st.subheader("Parâmetros Otimizados")
        st.json(metadata.get('best_params', {}))

    st.title(f"🤖 Sinalizador IA para {metadata.get('ativo', 'Ativo')}")

    col1, col2, col3 = st.columns(3)
    placeholder_sinal = col1.empty()
    placeholder_confianca = col2.empty()
    placeholder_preco = col3.empty()
    
    placeholder_status = st.empty()

    while True:
        placeholder_status.info("Buscando novos dados e a gerar previsão...")

        df_raw = buscar_dados_recentes(client, metadata['ativo'], metadata['intervalo'])
        
        if not df_raw.empty:
            df_features = gerar_features(df_raw)

            if not df_features.empty:
                last_row = df_features[FEATURES_LIST].iloc[[-1]]
                predicao = modelo.predict(last_row)[0]
                probabilidade = modelo.predict_proba(last_row)[0]

                sinal_texto = "🟢 ALTA" if predicao == 1 else "🔴 BAIXA"
                confianca = probabilidade[1] if predicao == 1 else probabilidade[0]
                preco_atual = df_raw['close'].iloc[-1]

                with placeholder_sinal:
                    st.metric("Sinal para o Próximo Candle", sinal_texto)
                with placeholder_confianca:
                    st.metric("Confiança do Modelo", f"{confianca * 100:.2f}%")
                with placeholder_preco:
                    st.metric(f"Preço Atual ({metadata['ativo']})", f"${preco_atual:,.4f}")

                agora = datetime.datetime.now().strftime("%H:%M:%S")
                placeholder_status.success(f"Dashboard atualizado às {agora}.")
            
            else:
                placeholder_status.warning("Não foi possível gerar features com os dados atuais. A tentar novamente...")
        
        else:
            placeholder_status.error("Falha ao buscar dados da Binance. A tentar novamente em 60 segundos...")

        time.sleep(60)

if __name__ == "__main__":
    main()
