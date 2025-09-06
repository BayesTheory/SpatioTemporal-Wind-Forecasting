# 🌬️ SpatioTemporal Wind Forecasting

An advanced Deep Learning pipeline for spatio-temporal wind speed forecasting using ConvLSTM, PredRNN, and a state-of-the-art Transformer model (PredFormer Fac-T-S). This project handles the entire workflow from data preprocessing to model training, hyperparameter tuning, and evaluation.

![Project Banner](https://github.com/BayesTheory/SpatioTemporal-Wind-Forecasting/main/sua_imagem_combinada.png)
*(Note: Please update the image link to a valid one for it to display correctly)*

---

## ✨ Features

-   **Multi-Model Architecture**: Implements and compares multiple deep learning models against a classical baseline:
    -   `ARIMA` (Baseline)
    -   `FCNN` (Fully Connected Neural Network)
    -   `ConvLSTM` (Recurrent-Convolutional)
    -   `PredRNN` (Spatiotemporal LSTM)
    -   `PredFormer (Fac-T-S)` (State-of-the-art pure Transformer)
-   **End-to-End Pipeline**: Automates the entire workflow, including data loading, preprocessing, training, and evaluation.
-   **Advanced Training Techniques**: Utilizes modern methods for high performance and stability:
    -   Automatic Mixed Precision (AMP) for faster training.
    -   Gradient Clipping to prevent exploding gradients.
    -   Configurable Optimizers (`Adam`, `AdamW`) and Schedulers (`CosineAnnealing`, `ReduceLROnPlateau`).
-   **Experiment Tracking**: Fully integrated with **MLflow** for comprehensive logging of parameters, metrics, and visual artifacts.
-   **Rigorous Validation**: Employs **Time Series Cross-Validation** to ensure robust model evaluation and prevent data leakage.
-   **Centralized Configuration**: All experiments are managed through a single, easy-to-use `modelos.json` file.

---

## 📈 Performance Results

After a full cycle of training and hyperparameter optimization, **PredFormer (Fac-T-S)** emerged as the top-performing model on the test set, demonstrating superior accuracy.

| Rank | Model                 | Best RMSE              | Best MAE               |
| :--: | :-------------------- | :--------------------- | :--------------------- |
| 🏆 1ˢᵗ | **PredFormer (Fac-T-S)** | **1.30760** | **0.93202** |
| 🥈 2ⁿᵈ | PredRNN               | 1.33124                | 0.94050                |
| 🥉 3ʳᵈ | ConvLSTM              | 1.33959                | 0.94973                |
| 🏅 4ᵗʰ | FCNN                  | 1.86136                | 1.35341                |
| 🏅 5ᵗʰ | ARIMA                 | 2.57494                | 1.90345                |

---

## 🛠️ Technologies Used

-   Python 3.8+
-   PyTorch
-   MLflow
-   Xarray & NetCDF4
-   Scikit-learn
-   Pandas & NumPy
-   Einops

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

#### 1. Clone the Repository
```bash
git clone [https://github.com/seu-usuario/SpatioTemporal-Wind-Forecasting.git](https://github.com/seu-usuario/SpatioTemporal-Wind-Forecasting.git)
cd SpatioTemporal-Wind-Forecasting
```

#### 2. Create Environment and Install Dependencies
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

#### 3. Configure the Experiment
-   Open the `modelos.json` file.
-   Set `"ativo": true` for the models you wish to train.
-   Adjust hyperparameters as needed for your experiment.

#### 4. Run the Main Pipeline
```bash
python main.py
```

#### 5. Visualize Results with MLflow
-   In a new terminal (from the same project directory), launch the MLflow UI:
```bash
mlflow ui
```
-   Open your web browser and navigate to `http://127.0.0.1:5000` to view and compare your experiment runs.

---

## 🔮 Future Work

-   [ ] **Containerization:** Create a `Dockerfile` to package the environment and code for easy deployment.
-   [ ] **Inference API:** Develop a `FastAPI` script to serve the champion model (`.pth`) via a REST API.
-   [ ] **CI/CD Pipeline:** Automate retraining and deployment on a cloud platform (e.g., AWS, GCP, Azure).

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

