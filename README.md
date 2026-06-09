# Financial Performance Classification | Finansal Performans Sınıflandırması

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-black.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg)

## Project Overview | Proje Özeti

**EN:**  
This project aims to classify the **financial performance** of companies using machine learning models based on their financial indicators. The project encompasses end-to-end machine learning workflows, from data analysis and feature engineering to model training and high-performance API deployment using **Go** and **ONNX**.

**TR:**  
Bu proje, şirketlere ait finansal veriler üzerinden makine öğrenimi modelleri kullanarak **finansal performans sınıflandırması** yapmayı amaçlamaktadır. Proje, veri analizi, özellik mühendisliği (feature engineering), model eğitimi ve modelin **Go** ve **ONNX** kullanılarak yüksek performanslı bir API olarak sunulmasına kadar uçtan uca makine öğrenimi süreçlerini kapsamaktadır.

---

## Project Structure | Proje Yapısı

The repository is organized to separate source code, data, models, and API:

```text
.
├── api/             # Go-based high-performance API for model inference using ONNX
├── data/
│   ├── raw/         # Original dataset
│   └── processed/   # Preprocessed and split datasets (train/test sets)
├── lib/             # Required libraries (e.g., ONNX runtime bindings)
├── models/          # Trained models (.pkl, .onnx) and preprocessing config (.json)
├── notebooks/       # Jupyter Notebooks for data creation, EDA, and evaluation
├── src/             # Core Python scripts (training, evaluation, exporting)
├── pyproject.toml   # Python dependencies and project configuration (uv)
└── README.md        # Project documentation
```

---

## Data Analysis & Feature Engineering | Veri Analizi ve Özellik Mühendisliği

**EN:**  
- **Data Inspection:** Missing value check, feature distributions, and correlation analysis.
- **Engineered Features:** Extracted new domain-specific features to improve model performance:
  - EBITDA Margin
  - ROA (Return on Assets), ROE (Return on Equity)
  - CashFlow/Debt, DSCR (Debt Service Coverage Ratio)
  - FCF (Free Cash Flow), Inventory Turnover
  - DSO (Days Sales Outstanding), P/B Ratio
- **Preprocessing:** Standard scaling of numerical variables and Train/Test splits using `Company_ID`.

**TR:**  
- **Veri İnceleme:** Eksik değer kontrolü, değişken dağılımları ve korelasyon analizi.
- **Özellik Mühendisliği (Feature Engineering):** Model performansını artırmak için yeni finansal metrikler türetildi:
  - FAVÖK Marjı (EBITDA Margin)
  - ROA, ROE
  - Nakit Akışı / Borç, DSCR (Borç Servis Karşılama Oranı)
  - Serbest Nakit Akışı (FCF), Stok Devir Hızı
  - DSO (Ortalama Tahsilat Süresi), P/B Oranı (Piyasa/Defter Değeri)
- **Ön İşleme:** Sayısal değişkenlerin standartlaştırılması (Scaling) ve `Company_ID` üzerinden Eğitim/Test veri setlerinin ayrılması.

---

## Modeling & API Integration | Modelleme ve API Entegrasyonu

**EN:**  
Models are trained using Python (Scikit-Learn). To achieve robust and fast inference, the finalized Support Vector Machine (SVM) model is exported to **ONNX** format. A Go-based API (`api/`) serves this ONNX model, allowing high-performance, concurrent financial performance predictions.

**TR:**  
Modeller Python (Scikit-Learn) kullanılarak eğitilmiştir. Hızlı ve dayanıklı bir çıkarım (inference) mekanizması sağlamak için nihai Destek Vektör Makinesi (SVM) modeli **ONNX** formatına dönüştürülmüştür. Go tabanlı bir API (`api/`) bu ONNX modelini kullanarak yüksek performanslı ve eşzamanlı finansal performans tahminleri sunar.

---

## Technologies Used | Kullanılan Teknolojiler

- **Machine Learning & Data Science:** Python 3.x, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Model Deployment:** ONNX, ONNX Runtime
- **Backend API:** Go (Golang)
- **Dependency Management:** `uv` (Python), `go mod` (Go)

---

## How to Run | Nasıl Çalıştırılır?

### 1. Python Environment Setup (Data Science & Training)

This project uses `uv` for ultra-fast Python package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Run Jupyter to explore notebooks
jupyter notebook
```

### 2. Running the Go API

To run the inference API built with Go:

```bash
# Navigate to the API directory
cd api

# Run the Go application
go run main.go
```

*(Ensure you have ONNX Runtime libraries configured correctly in your system or utilizing the provided `lib/` directory).*
