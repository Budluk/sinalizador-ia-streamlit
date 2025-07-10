# -*- coding: utf-8 -*-
"""
SCRIPT DE TREINAMENTO E OTIMIZAÇÃO DE MODELO

Este script é responsável por:
1. Conectar-se à API da Binance de forma segura.
2. Baixar um histórico de dados de um criptoativo (ex: BTCUSDT).
3. Gerar um conjunto rico de features de análise técnica.
4. Treinar um modelo de Machine Learning (RandomForestClassifier).
5. Otimizar os hiperparâmetros do modelo usando GridSearchCV.
6. Salvar o modelo treinado (.pkl) e seus metadados (.json) para uso na aplicação principal.
"""

import pandas as pd
import numpy as np
import joblib
import datetime
import os
import json
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException, BinanceRequestException
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# --- CONSTANTES ---

# Lista de features que o modelo utilizará.
# É VITAL que esta lista seja IDÊNTICA à usada no app principal.
FEATURES_LIST = [
    'retorno', 'media9', 'media21', 'volume', 'volume_ma', 'volume_diff_ma', 'volume_ratio_ma', 'volume_anomalo',
    'range_vela', 'close_pos_pct', 'open_pos_pct', 'body_size', 'body_ratio_range',
    'upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio',
    'pressao_compra', 'pressao_venda',
    'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 'adx',
    'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bbw', 'bb_bbp', 'obv', 'mfi'
]

# --- FUNÇÕES ---

def conectar_cliente_binance(api_key, api_secret):
    """Conecta-se à API da Binance de forma segura."""
    print("INFO: Tentando conectar à API da Binance...")
    try:
        client = BinanceClient(api_key, api_secret)
        # Valida as credenciais fazendo uma chamada simples
        client.get_account_status()
        print("SUCESSO: Conexão com a Binance estabelecida.")
        return client
    except (BinanceAPIException, BinanceRequestException) as e:
        print(f"ERRO DE API: Falha ao conectar à Binance. Verifique suas chaves e conexão. Detalhes: {e}")
        return None
    except Exception as e:
        print(f"ERRO INESPERADO: Ocorreu um erro ao conectar à Binance: {e}")
        return None

def carregar_dados_historicos(client, symbol, interval, start_str):
    """Carrega dados históricos (klines) da Binance."""
    print(f"INFO: Carregando dados para {symbol} (Intervalo: {interval}, Desde: {start_str})...")
    if not client:
        print("ERRO: Cliente Binance não inicializado.")
        return pd.DataFrame()

    try:
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            print(f"AVISO: Nenhum dado foi retornado para {symbol}.")
            return pd.DataFrame()

        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore']
        df = pd.DataFrame(klines, columns=cols)

        # Seleciona e converte as colunas mais importantes
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df.dropna(inplace=True)

        print(f"SUCESSO: {len(df)} barras de dados carregadas para {symbol}.")
        return df

    except Exception as e:
        print(f"ERRO ao carregar dados históricos: {e}")
        return pd.DataFrame()

def gerar_features(df):
    """Gera features de análise técnica a partir de dados OHLCV."""
    print("INFO: Iniciando geração de features...")
    if df.empty or len(df) < 30:
        print("AVISO: Dados insuficientes para gerar features (mínimo de 30 barras).")
        return pd.DataFrame()

    df_copy = df.copy()
    epsilon = 1e-9  # Para evitar divisão por zero

    # Price Action e Médias
    df_copy['retorno'] = df_copy['close'].pct_change()
    df_copy['media9'] = ta.trend.sma_indicator(df_copy['close'], window=9)
    df_copy['media21'] = ta.trend.sma_indicator(df_copy['close'], window=21)

    # Volume
    df_copy['volume_ma'] = ta.trend.sma_indicator(df_copy['volume'], window=10)
    df_copy['volume_diff_ma'] = df_copy['volume'] - df_copy['volume_ma']
    df_copy['volume_ratio_ma'] = df_copy['volume'] / (df_copy['volume_ma'] + epsilon)
    df_copy['volume_anomalo'] = (df_copy['volume_ratio_ma'] > 1.5).astype(int)

    # Estrutura da Vela (Candle)
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

    # Indicadores Técnicos (usando a biblioteca 'ta')
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

    # --- VARIÁVEL ALVO (TARGET) ---
    # Prever se o próximo candle fechará em ALTA (1) ou não (0).
    df_copy['target'] = (df_copy['close'].shift(-1) > df_copy['close']).astype(int)

    # Remove linhas com NaN geradas pelos indicadores
    df_copy.dropna(inplace=True)
    
    print(f"SUCESSO: Geração de features concluída. {len(df_copy)} amostras válidas.")
    return df_copy

def main():
    """Função principal para orquestrar o processo de otimização."""
    print("--- INICIANDO SCRIPT DE OTIMIZAÇÃO DE MODELO ---")

    # 1. CARREGAR CREDENCIAIS DE FORMA SEGURA
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("ERRO CRÍTICO: As variáveis de ambiente BINANCE_API_KEY e BINANCE_API_SECRET não estão configuradas.")
        print("Configure-as no seu sistema para continuar.")
        return # Encerra a execução

    # 2. CONECTAR À BINANCE
    client = conectar_cliente_binance(api_key, api_secret)
    if not client:
        return

    # 3. COLETAR E PROCESSAR DADOS
    ativo = "BTCUSDT"
    intervalo = "1h"
    data_inicio = "1 Jan, 2023" # Período maior para um modelo mais robusto
    
    df_historico = carregar_dados_historicos(client, ativo, intervalo, data_inicio)
    if df_historico.empty:
        return

    df_features = gerar_features(df_historico)
    if df_features.empty:
        return

    # 4. PREPARAR DADOS PARA TREINAMENTO
    X = df_features[FEATURES_LIST]
    y = df_features['target']

    if len(X) < 100:
        print(f"AVISO: Apenas {len(X)} amostras disponíveis. O treinamento pode não ser eficaz.")
        return

    # Divisão temporal (shuffle=False é crucial para dados de série temporal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"INFO: Dados divididos. Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras.")

    # 5. OTIMIZAÇÃO DE HIPERPARÂMETROS (GridSearchCV)
    print("\nINFO: Iniciando otimização de hiperparâmetros...")
    
    # Grade de parâmetros reduzida para um treinamento mais rápido.
    # Para uma busca mais completa, adicione mais valores.
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15],
        'min_samples_leaf': [3, 5],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3, # Cross-validation com 3 folds
        scoring='accuracy',
        verbose=2,
        n_jobs=-1 # Usar todos os núcleos de CPU disponíveis
    )
    grid_search.fit(X_train, y_train)

    print("\nSUCESSO: Otimização concluída!")
    print(f"Melhores Parâmetros: {grid_search.best_params_}")
    print(f"Melhor Acurácia em Validação Cruzada: {grid_search.best_score_:.4f}")

    # 6. AVALIAR E SALVAR O MELHOR MODELO
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)

    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO (CONJUNTO DE TESTE) ---")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia Final no Teste: {final_accuracy:.4f}")

    # 7. SALVAR ARQUIVOS PARA A APLICAÇÃO
    model_filename = 'modelo_ia.pkl'
    metadata_filename = 'metadata_modelo.json'

    joblib.dump(best_model, model_filename)
    
    metadata = {
        'accuracy_teste': final_accuracy,
        'best_params': grid_search.best_params_,
        'ativo': ativo,
        'intervalo': intervalo,
        'data_treino': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_amostras_treino': len(df_features),
        'features_utilizadas': FEATURES_LIST
    }
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Modelo salvo como '{model_filename}'")
    print(f"✅ Metadados salvos como '{metadata_filename}'")
    print("\n--- SCRIPT FINALIZADO COM SUCESSO ---")


if __name__ == "__main__":
    main()
