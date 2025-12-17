💧 Next-Gen Groundwater Quality Intelligence System

An AI-powered groundwater quality monitoring and decision-support system built using FastAPI, Machine Learning, and Environmental Standards (BIS/WHO).
This system analyzes groundwater samples, detects anomalies, computes Water Quality Index (WQI), checks regulatory compliance, and generates executive summaries using Large Language Models.

📌 Project Overview

Groundwater is a critical source of drinking and irrigation water. This project provides a smart, ML-driven API to assess groundwater quality using physicochemical parameters and environmental intelligence.

The system combines:

Machine Learning models (Autoencoder, PCA, Clustering)

Rule-based regulatory checks

Water Quality Index (WQI)

AI-generated executive summaries

Interactive visualizations

It is designed for research, academic projects, environmental monitoring, and decision support systems.

🎯 Key Features

🔍 Sample-based groundwater quality analysis

🧠 ML-based anomaly detection (Autoencoder)

📊 Water Quality Index (WQI) calculation

⚖️ BIS & WHO drinking water compliance checks

📉 Groundwater trend visualization

🧾 AI-generated executive summaries (Groq LLM)

🌐 RESTful API with Swagger documentation

📈 On-demand plots (PNG responses)

🧪 Water Quality Parameters

The system supports the following parameters:

pH

Total Dissolved Solids (TDS)

Nitrate

Chloride

Total Hardness

Sulfate

Fluoride

Iron

Electrical Conductivity

🧠 Machine Learning Architecture
1️⃣ Data Processing

Missing value handling using median statistics

Feature scaling using StandardScaler

Dimensionality reduction using PCA

2️⃣ Models Used

Autoencoder (PyTorch)

Detects anomalous groundwater samples

Random Forest Cluster Emulator

Assigns groundwater samples to learned quality clusters

3️⃣ Water Quality Index (WQI)

Weighted WQI calculation based on standard environmental formulas

Categories:

Excellent

Good

Moderate

Poor / Unsafe

⚖️ Regulatory Standards

The system automatically validates results against:

Bureau of Indian Standards (IS 10500:2012)

World Health Organization (WHO) Guidelines

Critical exceedances (e.g., Nitrate, Fluoride, Iron, pH) are explicitly flagged.

🌐 API Endpoints
🔹 Root
GET /


Returns API status and available endpoints.

🔹 Mode 1: Location-Based Analysis
POST /mode1/location-analysis


Provides:

Location context

Aquifer & soil information

Groundwater level trends

AI-generated executive summary

🔹 Groundwater Trend Plot
GET /mode1/trend-plot


Returns a PNG visualization of groundwater depletion trends.

🔹 Mode 2: Sample-Based Analysis
POST /mode2/sample-analysis


Returns:

ML anomaly detection results

WQI score & category

Regulatory compliance

Confidence estimation

AI-generated explanation

🔹 Sample Parameter Plot
POST /mode2/sample-plot


Returns a bar chart comparing parameters with BIS/WHO limits.

🔹 Summary Endpoint
GET /ai-summary


Returns a brief system description (for backward compatibility).

🛠️ Technologies Used

Python

FastAPI

PyTorch

Scikit-learn

Pandas & NumPy

Matplotlib

Groq LLM API

Uvicorn

📂 Project Structure
├── app.py                 # FastAPI backend
├── requirements.txt       # Python dependencies
├── main.ipynb             # Data analysis & model training
├── .gitignore             # Security & ignore rules
├── README.md              # Project documentation


⚠️ Trained ML models, datasets, and environment variables are excluded for security and best practices.

🚀 How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Set Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here
USGS_API_KEY=optional
EARTHDATA_BEARER_TOKEN=optional

3️⃣ Start the API
uvicorn app:app --reload

4️⃣ Open Swagger UI
http://127.0.0.1:8000/docs

🌍 Use Cases

Drinking water quality assessment

Groundwater contamination monitoring

Smart water management systems

Environmental impact analysis

Academic & research projects

🏆 Project Level

Advanced | Machine Learning | Environmental Intelligence

Suitable for:

Final-year engineering projects

Research demonstrations

ML & data science portfolios

Environmental analytics platforms

👨‍💻 Author

Shivendra Pandey
Computer Science Engineering
Aspiring Machine Learning Engineer

📜 Disclaimer

This system provides decision-support insights based on machine learning and illustrative data.
It should be used alongside laboratory testing for critical applications.