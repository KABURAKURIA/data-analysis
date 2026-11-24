# 📊 Ultimate Data Science & Social Science Research Analytics Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Analysis](https://img.shields.io/badge/Domain-Social%20Science%20%26%20NLP-success.svg)](https://github.com/KABURAKURIA/data-analysis)

A comprehensive, production-grade repository containing **100+ end-to-end analytical notebooks** bridging computational social science, natural language processing (NLP), statistical modeling, network analysis, and geospatial research.

---

## 🎯 Repository Overview

This repository provides reproducible research pipelines and empirical analytics methodologies designed for researchers, data scientists, and quantitative analysts:

- **💬 Natural Language Processing & Discourse Analysis**: WhatsApp chat forensics, sentiment analysis, Latent Dirichlet Allocation (LDA) topic modeling, n-grams, named-entity recognition (NER), and hate-speech detection.
- **📈 Advanced Statistical & Psychometric Modeling**: Survey data cleaning, Likert-scale psychometrics, ANOVA / hypothesis testing, linear and logistic regression, exploratory factor analysis (EFA), and principal component analysis (PCA).
- **🕸️ Network Science & Topology**: Graph construction, centrality metrics (Betweenness, Closeness, Eigenvector), and Louvain community detection.
- **🗺️ Spatial & Temporal Analytics**: Time-series forecasting, geospatial mapping, automated text coding, and inter-rater reliability scoring (Cohen's Kappa, Krippendorff's Alpha).

---

## 📂 Core Analytics Notebook Catalog

| Category | Notebook ID & Name | Analytical Methodologies |
| :--- | :--- | :--- |
| **NLP & Chat Mining** | `01_whatsapp_chat_analysis.ipynb` | Regex parsing, temporal message distributions, user interaction metrics |
| **Sentiment Analytics** | `02_sentiment_analysis.ipynb` | VADER, TextBlob, Transformer embeddings, polarity scoring |
| **Topic Modeling** | `03_topic_modeling.ipynb` | LDA, BERTopic, coherence score optimization |
| **Lexical Analysis** | `04_wordcloud_ngrams.ipynb` | TF-IDF, collocations, bi-gram/tri-gram extraction |
| **Information Extraction** | `05_named_entity_recognition.ipynb` | SpaCy NER, entity linking, entity co-occurrence networks |
| **Content Moderation** | `06_hate_speech_detection.ipynb` | Supervised text classification, toxic phrase identification |
| **Data Munging** | `07_text_preprocessing.ipynb` & `08_survey_cleaning.ipynb` | Tokenization, lemmatization, missing data imputation, outlier filtering |
| **Exploratory Stats** | `09_descriptive_stats.ipynb` & `10_likert_scale_analysis.ipynb` | Central tendency, skewness, Likert diverging stacked bar charts |
| **Inferential Analytics** | `11_group_comparisons.ipynb` & `12_linear_regression.ipynb` | t-tests, Mann-Whitney U, OLS regression, heteroscedasticity diagnostics |
| **Categorical Modeling** | `13_logistic_regression.ipynb` | Binary/Multinomial logit, odds ratios, ROC-AUC curves |
| **Dimensionality Reduction**| `14_factor_analysis.ipynb` & `15_cluster_analysis.ipynb` | Scree plots, Varimax rotation, K-Means, DBSCAN clustering |
| **Network Analysis** | `16_network_construction.ipynb` - `18_community_detection.ipynb` | NetworkX, centrality distributions, Louvain modularity |
| **Forecasting & GIS** | `19_time_series_analysis.ipynb` & `20_geospatial_mapping.ipynb` | ARIMA, seasonal decomposition, GeoPandas, choropleth maps |
| **Research Validation** | `21_auto_coding.ipynb` - `24_api_data_collection.ipynb` | Inter-rater reliability, RESTful API scraping, data pipelines |
| **Capstone Project** | `100_capstone_project.ipynb` | End-to-end multi-modal research synthesis project |

---

## 🚀 Quickstart & Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/KABURAKURIA/data-analysis.git
cd data-analysis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install analytical dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn statsmodels nltk spacy networkx geopandas jupyter

# 4. Launch Jupyter Notebook
jupyter notebook
```

---

## 🔬 Analytical Stack

- **Data Processing**: Pandas, NumPy, SciPy
- **Statistical Modeling**: Statsmodels, Scikit-Learn
- **Natural Language Processing**: NLTK, SpaCy, Gensim
- **Graph & Network Science**: NetworkX
- **Visualization**: Matplotlib, Seaborn, Altair, Plotly

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
