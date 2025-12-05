
#  Multi-Modal Automated Algorithmic Trading System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/NLP-FinBERT-yellow?style=for-the-badge&logo=huggingface)
![Google Colab](https://img.shields.io/badge/Compute-Google_Colab_GPU-blue?style=for-the-badge&logo=googlecolab)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A Final Year Project (FYP) utilizing a Hybrid Deep Learning Architecture (LSTM + Transformer-based FinBERT) to predict cryptocurrency market trends by fusing Technical (OHLCV) and Fundamental (News Sentiment) data.**

---

##  Project Team

| Role | Name | Key Responsibilities |
| :--- | :--- | :--- |
| **Supervisor** | **Sir Muhammad Adeel Zahid** | Project Guidance & Evaluation |
| **Data Scientist** | **Haider Ali** | Model Architecture, Pipeline Design, Trading Logic & Streamlit |
| **Data Scientist** | **Nimrah Akbar** | Data Preprocessing, Sentiment Analysis, Optimization & Backtesting |

---

##  Table of Contents
1.  [Executive Summary](#-executive-summary)
2.  [System Architecture](#-system-architecture)
3.  [Data Science Methodology](#-data-science-methodology)
4.  [Technology Stack](#-technology-stack)
5.  [Directory Structure](#-directory-structure)
6.  [Installation & Setup](#-installation--setup)
7.  [Project Roadmap](#-project-roadmap)
8.  [Disclaimer](#-disclaimer)

---

##  Executive Summary
The **Multi-Modal Crypto Trading Bot** is a high-frequency algorithmic trading system designed to address the stochastic nature of cryptocurrency markets. Traditional quantitative models often fail because they rely solely on historical price data (Technical Analysis), ignoring the massive impact of social sentiment and news events (Fundamental Analysis).

**Our Solution:**
We propose a **Multi-Modal Fusion Model** that processes two distinct data streams:
1.  **Time-Series Data:** Analyzed using **Long Short-Term Memory (LSTM)** networks to capture temporal dependencies in price action.
2.  **Unstructured Text Data:** Analyzed using **FinBERT** (Financial Bidirectional Encoder Representations from Transformers) to extract sentiment polarity from news.

The system outputs a unified **Confidence Score** for Buy/Sell signals, executed via a simulated trading engine with strict risk management protocols (RSI & Stop-Loss).

---

##  System Architecture

The project follows a standard **ETL (Extract, Transform, Load)** pipeline integrated with an Inference Engine.

```mermaid
graph TD;
    subgraph "Data Ingestion Layer"
    A[Yahoo Finance API] -->|Raw OHLCV| C(Data Preprocessing);
    B[Kaggle & RSS Feeds] -->|Raw Text| C;
    C -->|MinMax Normalization| D[Cleaned Data Store];
    end
    
    subgraph "Deep Learning Layer"
    D -->|Seq Length 60| E[LSTM Model];
    D -->|Tokenization| F[FinBERT Model];
    end
    
    subgraph "Decision Layer"
    E -->|Price Prediction| G{Fusion Engine};
    F -->|Sentiment Score| G;
    G -->|Trade Logic| H[Trade Execution];
    end
    
    subgraph "Presentation Layer"
    H -->|Metrics| I[Streamlit Dashboard];
    end
````

-----

## 🔬 Data Science Methodology

### 1\. Technical Analysis (The LSTM Model)

We utilize **Recurrent Neural Networks (RNNs)**, specifically **Stack LSTMs**, to model the sequential nature of financial data.

  * **Problem Type:** Many-to-One Regression.
  * **Input Features:** Open, High, Low, Close, Volume.
  * **Lookback Window:** 60 Days (The model looks at the past 60 days to predict t+1).
  * **Optimization:** Adam Optimizer with Mean Squared Error (MSE) loss function.

### 2\. Fundamental Analysis (The FinBERT Model)

Instead of using basic dictionary-based approaches (like VADER), we implement **Transfer Learning** using FinBERT.

  * **Architecture:** BERT-base model fine-tuned on financial corpus (TRC2).
  * **Output:** Softmax probabilities for three classes: `[Positive, Negative, Neutral]`.

### 3\. Quantitative Strategy (The Alpha Logic)

The trading signal is generated based on a **Confluence Strategy**:
$$ Signal = \begin{cases} BUY, & \text{if } P_{pred} > P_{curr} \times 1.01 \text{ AND } S_{score} > 0.2 \text{ AND } RSI < 70 \\ SELL, & \text{if } P_{pred} < P_{curr} \times 0.99 \text{ OR } S_{score} < -0.5 \\ HOLD, & \text{otherwise} \end{cases} $$

-----

##  Technology Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Environment** | Google Colab | Utilizes free T4 GPU for faster model training. |
| **Core** | Python 3.10 | Standard language for Data Science & AI. |
| **Deep Learning** | TensorFlow / Keras | High-level API for rapid prototyping of LSTMs. |
| **NLP** | HuggingFace Transformers | State-of-the-art pre-trained models (FinBERT). |
| **Data Manipulation** | Pandas & NumPy | Vectorized operations for large datasets. |
| **Visualization** | Streamlit & Matplotlib | Interactive dashboards for end-users. |
| **Version Control** | Git & GitHub | Collaborative development. |

-----

##  Directory Structure

```text
📦 FYP-Crypto-Trading-Bot
 ┣ 📂 datasets           # Raw & Processed Data (Managed via Google Drive)
 ┣ 📂 models             # Serialized Models (.h5 for LSTM, .pt for BERT)
 ┣ 📂 notebooks          # Experimental Jupyter Notebooks
 ┃ ┣ 📜 01_Data_Preprocessing.ipynb   # Cleaning & Feature Engineering
 ┃ ┣ 📜 02_Model_Training.ipynb       # Model Training Loop
 ┃ ┗ 📜 03_Backtesting_Engine.ipynb   # Strategy Simulation
 ┣ 📂 src                # Production Scripts
 ┃ ┣ 📜 data_loader.py   # Yahoo Finance Fetcher
 ┃ ┗ 📜 indicators.py    # RSI/MACD Logic
 ┣ 📜 requirements.txt   # Dependencies
 ┗ 📜 README.md          # Documentation
```

-----

##  Installation & Setup

Since this project requires GPU acceleration, we highly recommend running the notebooks on **Google Colab**.

### Step 1: Clone Repository

```bash
git clone https://github.com/haiderali-01/FYP-Crypto-Trading-Bot.git
```

### Step 2: Configure Data Layer (Google Drive)

The bot requires a persistent storage layer to save datasets and trained models.

1.  Create a folder in your Google Drive named: **`FYP DATAETS`**.
2.  Upload the `news_data.csv` (downloaded from Kaggle) into this folder.
3.  *Note:* Market price data is auto-fetched via our `data_loader.py` script.

### Step 3: Execution

Open the notebooks in the `notebooks/` folder in the following order:

1.  **`01_Data_Preprocessing`**: Syncs Price and News data.
2.  **`02_Model_Training`**: Trains the AI models.
3.  **`03_Backtesting_Engine`**: Simulates the trading strategy.

-----

##  Project Roadmap

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Data Engineering:** ETL Pipelines, Feature Engineering (RSI, MACD). | 🟡 In Progress |
| **Phase 2** | **Model Development:** LSTM Training & FinBERT Integration. | ⚪ Planned |
| **Phase 3** | **Strategy Formulation:** Developing the Hybrid Decision Engine. | ⚪ Planned |
| **Phase 4** | **Validation:** Backtesting on 2024 unseen data & Performance Metrics. | ⚪ Planned |
| **Phase 5** | **Deployment:** Streamlit Dashboard & Final Reporting. | ⚪ Planned |

-----

##  Disclaimer & Liability Policy

This software is engineered as a **robust, real-time algorithmic trading system** capable of executing live market strategies. While it demonstrates high-performance capabilities during backtesting and simulation, it is primarily developed for **academic research** as a Data Science Final Year Project.

**Important Note for Users:**
1.  **Not Financial Advice:** The signals generated by this bot are based on probabilistic AI models (LSTM/FinBERT). They should not be taken as guaranteed financial advice.
2.  **Market Risk:** Cryptocurrency markets are highly volatile. The authors (**Haider Ali & Nimrah Akbar**) and the Supervisor are not liable for any financial losses incurred while using this software in a live production environment.
3.  **Use Responsibly:** We recommend running this system in **Paper Trading Mode** (Simulation) before deploying real capital.

---

-----

````