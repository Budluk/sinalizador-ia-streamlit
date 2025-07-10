# -*- coding: utf-8 -*-
"""
Script para otimização de um modelo de Machine Learning para prever movimentos de preços
de criptoativos, utilizando dados históricos da Binance.

Este script realiza os seguintes passos:
1. Conecta-se à API da Binance.
2. Baixa dados históricos (klines/velas) para um ativo específico.
3. Gera um conjunto rico de features de análise técnica e price action.
4. Divide os dados em conjuntos de treino e teste.
5. Utiliza GridSearchCV para encontrar os melhores hiperparâmetros para um modelo RandomForestClassifier.
6. Avalia o modelo otimizado no conjunto de teste.
7. Salva o melhor modelo (.pkl) e sua acurácia (.json) para uso posterior.
"""

import pandas as pd
import numpy as np
import joblib
import datetime
import os
import json

# --- Importa a biblioteca Binance ---
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

# --- Importa as bibliotecas de análise técnica (ta) ---
import ta.momentum as tam
import ta.volatility as tav
import ta.trend as tat
import ta.volume as tavol

# --- Importa componentes do Scikit-learn ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# --- Constante Global para Features ---
# Manter esta lista consistente entre os scripts de treino e de aplicação é crucial.
FEATURES_LIST = [
    'retorno', 'media9', 'media21', 'volume', 'volume_ma', 'volume_diff_ma', 'volume_ratio_ma', 'volume_anomalo',
    'range_vela', 'close_pos_pct', 'open_pos_pct', 'body_size', 'body_ratio_range',
    'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio',
    'pressao_compra', 'pressao_venda',
    'rsi', 
    'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 'adx', 
    'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbw', 'bb_bbp', 'obv', 'mfi'
]

# --- Funções de Conexão e Coleta de Dados ---

def connect_binance_client(api_key, api_secret):
    """
    Conecta-se à API da Binance e verifica a validade das credenciais.
    """
    try:
        client = BinanceClient(api_key, api_secret)
        client.get_account_status()  # Chamada para validar as credenciais
        print("INFO: Conexão com a API da Binance bem-sucedida.")
        return client
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"ERRO: Falha ao conectar à Binance. Verifique suas chaves de API e conexão. Detalhes: {e}")
        return None
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado ao conectar à Binance: {e}")
        return None

def carregar_dados_binance(client, symbol, interval, start_str, limit=None):
    """
    Carrega dados históricos de klines da Binance a partir de uma data específica.
    """
    print(f"INFO: Carregando dados para {symbol}, Intervalo: {interval}, Desde: {start_str}...")
    if not client:
        print("ERRO: Cliente Binance não está conectado. Impossível carregar dados.")
        return pd.DataFrame()

    try:
        # A API permite buscar até 1000 klines por chamada. Para períodos mais longos,
        # seria necessário implementar um loop de paginação.
        klines = client.get_historical_klines(symbol, interval, start_str, limit=limit)
        
        if not klines:
            print(f"AVISO: Nenhum dado retornado para {symbol} com os parâmetros fornecidos.")
            return pd.DataFrame()

        # Define as colunas com base na documentação da API da Binance
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(klines, columns=cols)
        
        # --- Limpeza e Formatação dos Dados ---
        # Seleciona colunas de interesse
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Converte colunas para o tipo numérico correto
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Converte o tempo de abertura para Datetime e define como índice
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        df.dropna(inplace=True)
        
        print(f"INFO: Dados carregados com sucesso. Total de {len(df)} barras.")
        return df

    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"ERRO: Erro na API da Binance ao buscar dados: {e}. Verifique o símbolo '{symbol}' e o intervalo.")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERRO: Erro inesperado ao carregar dados: {e}")
        return pd.DataFrame()


# --- Função de Geração de Features ---

def gerar_features(df):
    """
    Gera um conjunto de features de análise técnica e price action a partir de dados OHLCV.
    Versão otimizada para legibilidade e eficiência.
    """
    if df.empty or len(df) < 30:
        print("AVISO: DataFrame vazio ou com menos de 30 barras. Features não podem ser geradas.")
        return pd.DataFrame()

    df_copy = df.copy()
    epsilon = 1e-9 # Valor pequeno para evitar divisão por zero

    # --- Features Básicas e Médias Móveis ---
    df_copy['retorno'] = df_copy['close'].pct_change()
    df_copy['media9'] = df_copy['close'].rolling(9).mean()
    df_copy['media21'] = df_copy['close'].rolling(21).mean()

    # --- Features de Volume ---
    df_copy['volume_ma'] = df_copy['volume'].rolling(10).mean()
    df_copy['volume_diff_ma'] = df_copy['volume'] - df_copy['volume_ma']
    df_copy['volume_ratio_ma'] = df_copy['volume'] / (df_copy['volume_ma'] + epsilon)
    df_copy['volume_anomalo'] = (df_copy['volume_ratio_ma'] > 1.5).astype(int)

    # --- Features de Price Action (Velas) ---
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

    # --- Indicadores Técnicos (Biblioteca TA) ---
    df_copy['rsi'] = tam.RSIIndicator(df_copy['close'], window=14).rsi()
    stoch = tam.StochasticOscillator(df_copy['high'], df_copy['low'], df_copy['close'], window=14, smooth_window=3)
    df_copy['stoch_k'] = stoch.stoch()
    df_copy['stoch_d'] = stoch.stoch_signal()
    
    macd = tat.MACD(df_copy['close'], window_fast=12, window_slow=26, window_sign=9)
    df_copy['macd'] = macd.macd()
    df_copy['macd_signal'] = macd.macd_signal()
    df_copy['macd_diff'] = macd.macd_diff()
    
    df_copy['adx'] = tat.ADXIndicator(df_copy['high'], df_copy['low'], df_copy['close'], window=14).adx()

    bollinger = tav.BollingerBands(df_copy['close'], window=20, window_dev=2)
    df_copy['bb_bbm'] = bollinger.bollinger_mavg()
    df_copy['bb_bbh'] = bollinger.bollinger_hband()
    df_copy['bb_bbl'] = bollinger.bollinger_lband()
    df_copy['bb_bbw'] = bollinger.bollinger_wband()
    df_copy['bb_bbp'] = bollinger.bollinger_percentb()
    
    df_copy['atr'] = tav.AverageTrueRange(df_copy['high'], df_copy['low'], df_copy['close'], window=14).average_true_range()
    
    df_copy['obv'] = tavol.OnBalanceVolumeIndicator(df_copy['close'], df_copy['volume']).on_balance_volume()
    df_copy['mfi'] = tavol.MoneyFlowIndex(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=14).money_flow_index()

    # --- Variável Alvo (Target) ---
    # O alvo é prever se o próximo candle fechará em alta (1) ou não (0).
    df_copy['target'] = (df_copy['close'].shift(-1) > df_copy['close']).astype(int)

    # --- Limpeza Final ---
    # Remove linhas com valores NaN que foram gerados pelos indicadores com janelas
    df_copy.dropna(inplace=True)
    
    print("INFO: Geração de features concluída.")
    return df_copy


# --- Script Principal de Otimização ---
if __name__ == "__main__":
    print("--- Iniciando Script de Otimização de Modelo ---")

    # Carrega as credenciais da Binance a partir das variáveis de ambiente.
    # É a forma mais segura de gerenciar chaves, evitando expô-las no código.
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("ERRO CRÍTICO: Variáveis de ambiente BINANCE_API_KEY e BINANCE_API_SECRET não configuradas.")
        print("Por favor, configure as variáveis de ambiente para executar o script.")
        exit()

    # 1. Conectar à Binance
    client = connect_binance_client(api_key, api_secret)
    if not client:
        print("Finalizando o script devido à falha na conexão.")
        exit()

    # 2. Definir parâmetros para coleta de dados
    ativo_para_otimizacao = "BTCUSDT"
    intervalo_historico = "1h"
    start_date_optimize = "1 Jan, 2023" # Usar um período maior para ter mais dados

    df_historico = carregar_dados_binance(client, ativo_para_otimizacao, intervalo_historico, start_date_optimize)

    if df_historico.empty:
        print("Finalizando o script: não foi possível carregar dados históricos.")
        exit()

    # 3. Gerar features
    df_features = gerar_features(df_historico)

    if df_features.empty:
        print("Finalizando o script: não foi possível gerar features.")
        exit()

    # 4. Preparar dados para o modelo (X e y)
    X = df_features[FEATURES_LIST]
    y = df_features['target']

    if len(X) < 100:
        print(f"AVISO: Dados insuficientes ({len(X)} amostras) para um treinamento robusto. Recomenda-se pelo menos 100.")
        exit()

    # Divide os dados mantendo a ordem temporal (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Tamanho do conjunto de treino: {len(X_train)}, Teste: {len(X_test)}")

    # 5. Otimização de Hiperparâmetros com GridSearchCV
    print("\nIniciando otimização de hiperparâmetros (GridSearchCV)...")
    
    # Grade de parâmetros para testar. Ajustado para ser mais rápido.
    # Para uma busca mais exaustiva, aumente as opções e o valor de 'cv'.
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'class_weight': ['balanced', None]
    }

    # Configura o GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3, # Validação cruzada com 3 folds (bom equilíbrio para séries temporais)
        scoring='accuracy',
        verbose=2,
        n_jobs=-1 # Usa todos os processadores disponíveis
    )

    grid_search.fit(X_train, y_train)

    print("\nOtimização concluída!")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor score de acurácia (validação cruzada): {grid_search.best_score_:.4f}")

    # 6. Avaliar o melhor modelo no conjunto de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n--- Relatório de Classificação no Conjunto de Teste ---")
    print(classification_report(y_test, y_pred))
    final_accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia final no teste: {final_accuracy:.4f}")

    # 7. Salvar o modelo e os metadados
    model_filename = 'modelo_otimizado.pkl'
    metadata_filename = 'modelo_otimizado_metadata.json'

    joblib.dump(best_model, model_filename)
    
    metadata = {
        'accuracy_teste': final_accuracy,
        'best_params': grid_search.best_params_,
        'ativo': ativo_para_otimizacao,
        'intervalo': intervalo_historico,
        'data_treino': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features_utilizadas': FEATURES_LIST
    }
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Modelo otimizado salvo como '{model_filename}'")
    print(f"✅ Metadados do modelo salvos como '{metadata_filename}'")
    print("\n--- Script concluído com sucesso! ---")
