# 🌬️ SpatioTemporal Wind Forecasting

An advanced Deep Learning pipeline for spatio-temporal wind speed forecasting using ConvLSTM, PredRNN, and a state-of-the-art Transformer model (PredFormer Fac-T-S). This project handles the entire workflow from data preprocessing to model training, hyperparameter tuning, and evaluation.

---

## ✨ Features

* **Multi-Model Architecture:** Implements and compares multiple deep learning models:
    * `ARIMA` (Baseline)
    * `FCNN` (Fully Connected Neural Networks)
    * `ConvLSTM` (Recurrent-Convolutional)
    * `PredRNN` (Spatiotemporal LSTM)
    * `PredFormer (Fac-T-S)` (State-of-the-art pure Transformer)
* **End-to-End Pipeline:** Automates data loading, preprocessing, training, and evaluation.
* **Advanced Training:** Utilizes modern techniques for high performance:
    * Automatic Mixed Precision (AMP) for faster training.
    * Gradient Clipping for stability.
    * Configurable Optimizers (`Adam`, `AdamW`) and Schedulers (`CosineAnnealing`, `ReduceLROnPlateau`).
* **Experiment Tracking:** Fully integrated with **MLflow** for logging parameters, metrics, and visual artifacts.
* **Rigorous Validation:** Employs Time Series Cross-Validation to ensure robust model evaluation and prevent data leakage.
* **Centralized Configuration:** All experiments are controlled via a single `modelos.json` file.

## 🛠️ Tecnologias Utilizadas

* Python 3.8+
* PyTorch
* MLflow
* Xarray & NetCDF4
* Scikit-learn
* Pandas & NumPy
* Einops

## 🚀 Como Rodar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/SpatioTemporal-Wind-Forecasting.git](https://github.com/seu-usuario/SpatioTemporal-Wind-Forecasting.git)
    cd SpatioTemporal-Wind-Forecasting
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure o Experimento:**
    * Abra o arquivo `modelos.json`.
    * Defina `"ativo": true` para os modelos que você deseja treinar.
    * Ajuste os hiperparâmetros conforme necessário.

4.  **Execute o Pipeline Principal:**
    ```bash
    python main.py
    ```

5.  **Visualize os Resultados com MLflow:**
    * Em um novo terminal, na mesma pasta, execute:
    ```bash
    mlflow ui
    ```
    * Acesse `http://127.0.0.1:5000` no seu navegador.

## 📈 Resultados Finais

Após um ciclo completo de otimização, o `PredFormer` se tornou o modelo com o melhor desempenho no conjunto de teste:

| Posição | Modelo | Melhor RMSE | Melhor MAE |
| :--- | :--- | :--- | :--- |
| 🏆 **1º** | **5predformer (Fac-T-S)** | **1.30760** | **0.93202** |
| 🥈 2º | 4predrnn | 1.33124 | 0.94050 |
| 🥉 3º | 3convlstm | 1.33959 | 0.94973 |
| 🏅 4º | 2fcnn | 1.86136 | 1.35341 |
| 🏅 5º | 1arima | 2.57494 | 1.90345 |

## 🔮 Próximos Passos

* [ ] **Containerização:** Criar um `Dockerfile` para empacotar o ambiente e o código.
* [ ] **API de Inferência:** Desenvolver um script com FastAPI para servir o modelo campeão (`.pth`) via API.
* [ ] **Pipeline End-to-End (CI/CD):** Automatizar o retreino e deploy em uma plataforma de nuvem (AWS, GCP, Azure).

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

