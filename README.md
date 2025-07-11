🤖 Sinalizador de Daytrade com IA para Criptomoedas

Este projeto é uma aplicação web construída com Streamlit que utiliza um modelo de Machine Learning (RandomForest) para prever a direção do próximo candle de um ativo de criptomoeda (especificamente BTCUSDT). A aplicação busca dados em tempo real da Binance, gera features de análise técnica e exibe um sinal de "ALTA" ou "BAIXA" com a confiança do modelo.



Aceda à aplicação online aqui!



🚀 Funcionalidades

Dashboard em Tempo Real: A interface atualiza automaticamente a cada 60 segundos com a previsão mais recente.



Previsão com IA: Utiliza um modelo RandomForestClassifier treinado para prever se o próximo candle será de alta ou baixa.



Métricas Claras: Exibe o sinal (🟢 ALTA / 🔴 BAIXA), a confiança da previsão e o preço atual do ativo.



Informações do Modelo: Mostra na barra lateral detalhes sobre o modelo em uso, como a sua acurácia, data de treino e os melhores hiperparâmetros encontrados.



Script de Treino Separado: Inclui um script (optimize\_model.py) para que qualquer pessoa possa treinar e otimizar o seu próprio modelo.



🛠️ Como Executar

Existem duas formas de executar este projeto: online ou localmente.



1\. Online (Streamlit Cloud)

A forma mais fácil é aceder à versão já implementada da aplicação através deste link:



https://sinalizador-ia-app-crqrtfmohrkh6ij8zzshhz.streamlit.app/



2\. Localmente

Se preferir executar na sua própria máquina, siga estes passos:



Pré-requisitos:



Python 3.9+



Git



Miniconda (ou outro gestor de ambientes Python)



Passos:



Clone o repositório:



git clone https://github.com/Budluk/sinalizador-ia-streamlit.git

cd sinalizador-ia-streamlit



Crie e ative um ambiente virtual:



conda create --name sinalizador\_env python=3.9

conda activate sinalizador\_env



Instale as dependências:



pip install -r requirements.txt



Configure as suas chaves de API:



Crie uma pasta chamada .streamlit dentro da pasta do projeto.



Dentro da pasta .streamlit, crie um ficheiro chamado secrets.toml.



Adicione as suas chaves da Binance ao ficheiro secrets.toml neste formato:



\[binance]

BINANCE\_API\_KEY = "SUA\_CHAVE\_API\_AQUI"

BINANCE\_API\_SECRET = "SEU\_SEGREDO\_API\_AQUI"



Execute a aplicação Streamlit:



streamlit run daytrade\_app\_completo.py



📁 Estrutura do Projeto

.

├── .streamlit/

│   └── secrets.toml      # Ficheiro para guardar as chaves de API (localmente)

├── .gitignore            # Ficheiro que especifica o que o Git deve ignorar

├── daytrade\_app\_completo.py # O código principal da aplicação Streamlit

├── optimize\_model.py     # Script para treinar e otimizar o modelo de IA

├── requirements.txt      # Lista de dependências Python do projeto

├── modelo\_ia.pkl         # Ficheiro do modelo treinado

├── metadata\_modelo.json  # Metadados sobre o treino do modelo

└── README.md             # Este ficheiro



⚖️ Aviso Legal

Este projeto foi desenvolvido para fins educacionais e de estudo. As previsões geradas pela IA não são recomendações financeiras. O mercado de criptomoedas é extremamente volátil. Não utilize este projeto como a única base para as suas decisões de investimento. Faça a sua própria pesquisa e gestão de risco.

