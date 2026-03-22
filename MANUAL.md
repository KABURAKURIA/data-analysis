# Social Science Research Notebooks Manual

This manual provides an overview of the tools available in this repository. All tools have been verified as functional.

## Table of Contents

- [WhatsApp Chat Analysis In-Depth](#whatsapp-chat-analysis-in-depth)
- [Sentiment Analysis In-Depth](#sentiment-analysis-in-depth)
- [Topic Modeling (LDA) In-Depth](#topic-modeling-lda-in-depth)
- [Word Cloud & N-Grams In-Depth](#word-cloud-n-grams-in-depth)
- [Named Entity Recognition (NER) In-Depth](#named-entity-recognition-ner-in-depth)
- [Hate Speech / Toxicity Detection In-Depth](#hate-speech-toxicity-detection-in-depth)
- [Text Preprocessing Pipeline In-Depth](#text-preprocessing-pipeline-in-depth)
- [Survey Data Cleaning: An In-Depth Guide](#survey-data-cleaning:-an-in-depth-guide)
- [Descriptive Statistics In-Depth](#descriptive-statistics-in-depth)
- [Capstone Project: End-to-End Social Media Analysis](#capstone-project:-end-to-end-social-media-analysis)
- [Likert Scale Analysis In-Depth](#likert-scale-analysis-in-depth)
- [Group Comparisons: T-tests & ANOVA In-Depth](#group-comparisons:-t-tests-anova-in-depth)
- [Linear Regression Analysis In-Depth](#linear-regression-analysis-in-depth)
- [Logistic Regression In-Depth](#logistic-regression-in-depth)
- [Exploratory Factor Analysis (EFA) In-Depth](#exploratory-factor-analysis-efa-in-depth)
- [Cluster Analysis In-Depth](#cluster-analysis-in-depth)
- [Social Network Construction & Analysis In-Depth](#social-network-construction-analysis-in-depth)
- [Network Centrality Measures In-Depth](#network-centrality-measures-in-depth)
- [Community Detection In-Depth](#community-detection-in-depth)
- [Time Series Analysis In-Depth](#time-series-analysis-in-depth)
- [Geospatial Mapping In-Depth](#geospatial-mapping-in-depth)
- [Automated Coding Assistant In-Depth](#automated-coding-assistant-in-depth)
- [Inter-rater Reliability In-Depth](#inter-rater-reliability-in-depth)
- [Web Scraping for Social Science: An In-Depth Guide](#web-scraping-for-social-science:-an-in-depth-guide)
- [API Data Collection In-Depth](#api-data-collection-in-depth)
- [Data Format Converter In-Depth](#data-format-converter-in-depth)
- [Missing Data Imputation In-Depth](#missing-data-imputation-in-depth)
- [Merge and Join Datasets In-Depth](#merge-and-join-datasets-in-depth)
- [Outlier Detection In-Depth](#outlier-detection-in-depth)
- [Causal Inference: Propensity Score Matching (PSM) In-Depth](#causal-inference:-propensity-score-matching-psm-in-depth)
- [Model Evaluation Metrics In-Depth](#model-evaluation-metrics-in-depth)
- [Multilevel Modeling (Hierarchical Linear Models) In-Depth](#multilevel-modeling-hierarchical-linear-models-in-depth)
- [Survival Analysis In-Depth](#survival-analysis-in-depth)
- [Difference-in-Differences (DiD) In-Depth](#difference-in-differences-did-in-depth)
- [Regression Discontinuity Design (RDD) In-Depth](#regression-discontinuity-design-rdd-in-depth)
- [Fixed Effects Model (Panel Data) In-Depth](#fixed-effects-model-panel-data-in-depth)
- [Mediation Analysis In-Depth](#mediation-analysis-in-depth)
- [Moderation Analysis In-Depth](#moderation-analysis-in-depth)
- [Meta-Analysis Visualization In-Depth](#meta-analysis-visualization-in-depth)
- [Document Similarity In-Depth](#document-similarity-in-depth)
- [Supervised Text Classification In-Depth](#supervised-text-classification-in-depth)
- [Decision Tree Classifier In-Depth](#decision-tree-classifier-in-depth)
- [Random Forest Analysis In-Depth](#random-forest-analysis-in-depth)
- [K-Nearest Neighbors (KNN) In-Depth](#k-nearest-neighbors-knn-in-depth)
- [PDF Text Extraction In-Depth](#pdf-text-extraction-in-depth)
- [Nested JSON Flattening In-Depth](#nested-json-flattening-in-depth)
- [Advanced Date and Time Handling In-Depth](#advanced-date-and-time-handling-in-depth)
- [Interactive Widgets In-Depth](#interactive-widgets-in-depth)
- [Data Anonymization & Privacy In-Depth](#data-anonymization-privacy-in-depth)
- [Multiple Imputation for Missing Data In-Depth](#multiple-imputation-for-missing-data-in-depth)
- [Exporting Results In-Depth](#exporting-results-in-depth)
- [Repeated Measures ANOVA In-Depth](#repeated-measures-anova-in-depth)
- [Mixed ANOVA (Split-Plot Design) In-Depth](#mixed-anova-split-plot-design-in-depth)
- [Signal Detection Theory (SDT) In-Depth](#signal-detection-theory-sdt-in-depth)
- [Reaction Time Analysis In-Depth](#reaction-time-analysis-in-depth)
- [Power Analysis In-Depth](#power-analysis-in-depth)
- [Reliability Analysis (Cronbach's Alpha) In-Depth](#reliability-analysis-cronbach's-alpha-in-depth)
- [Effect Size Calculation In-Depth](#effect-size-calculation-in-depth)
- [Chi-Square Test of Independence In-Depth](#chi-square-test-of-independence-in-depth)
- [Mediation Analysis with Bootstrapping In-Depth](#mediation-analysis-with-bootstrapping-in-depth)
- [Plotting Interaction Effects In-Depth](#plotting-interaction-effects-in-depth)
- [Data Analyst Roadmap & Guide In-Depth](#data-analyst-roadmap-guide-in-depth)
- [SQL for Data Analysts In-Depth](#sql-for-data-analysts-in-depth)
- [Advanced Visualization with Plotly In-Depth](#advanced-visualization-with-plotly-in-depth)
- [Introduction to Transformers for NLP In-Depth](#introduction-to-transformers-for-nlp-in-depth)
- [Bayesian Analysis with PyMC In-Depth](#bayesian-analysis-with-pymc-in-depth)
- [Retrieval-Augmented Generation (RAG) for Research In-Depth](#retrieval-augmented-generation-rag-for-research-in-depth)
- [Building Interactive Dashboards with Panel In-Depth](#building-interactive-dashboards-with-panel-in-depth)
- [Causal Inference: Instrumental Variables (IV) In-Depth](#causal-inference:-instrumental-variables-iv-in-depth)
- [Python In-Depth for Social Scientists](#python-in-depth-for-social-scientists)
- [Pandas Manipulation In-Depth](#pandas-manipulation-in-depth)
- [Data Visualization In-Depth Guide](#data-visualization-in-depth-guide)
- [Advanced ML: Gradient Boosting with XGBoost In-Depth](#advanced-ml:-gradient-boosting-with-xgboost-in-depth)
- [Neural Networks for Tabular Data (PyTorch) In-Depth](#neural-networks-for-tabular-data-pytorch-in-depth)
- [Exploratory Data Analysis (EDA) In-Depth](#exploratory-data-analysis-eda-in-depth)
- [MLOps: Model Persistence In-Depth](#mlops:-model-persistence-in-depth)
- [Hypothesis Testing Roadmap In-Depth](#hypothesis-testing-roadmap-in-depth)
- [Audio Transcription with Whisper In-Depth](#audio-transcription-with-whisper-in-depth)
- [Image Analysis In-Depth](#image-analysis-in-depth)
- [Structural Equation Modeling (SEM) In-Depth](#structural-equation-modeling-sem-in-depth)
- [A/B Testing and Power Analysis In-Depth](#ab-testing-and-power-analysis-in-depth)
- [Conjoint Analysis (Rating-Based) In-Depth](#conjoint-analysis-rating-based-in-depth)
- [Algorithmic Fairness Audit In-Depth](#algorithmic-fairness-audit-in-depth)
- [Advanced Topic Modeling with BERTopic In-Depth](#advanced-topic-modeling-with-bertopic-in-depth)
- [Handling Large Datasets In-Depth](#handling-large-datasets-in-depth)
- [Multimodal Analysis: Text and Image In-Depth](#multimodal-analysis:-text-and-image-in-depth)
- [Recommender Systems In-Depth](#recommender-systems-in-depth)
- [Explainable AI (XAI) with SHAP In-Depth](#explainable-ai-xai-with-shap-in-depth)
- [Time Series Forecasting with Prophet In-Depth](#time-series-forecasting-with-prophet-in-depth)
- [Qualitative Analysis with LLMs (Zero-Shot Classification) In-Depth](#qualitative-analysis-with-llms-zero-shot-classification-in-depth)
- [Agentic AI Concepts: The ReAct Pattern In-Depth](#agentic-ai-concepts:-the-react-pattern-in-depth)
- [Handling Imbalanced Datasets (SMOTE & ADASYN) In-Depth](#handling-imbalanced-datasets-smote-adasyn-in-depth)
- [Bipartite Network Analysis (Affiliation Networks) In-Depth](#bipartite-network-analysis-affiliation-networks-in-depth)
- [Synthetic Control Methods (SCM) In-Depth](#synthetic-control-methods-scm-in-depth)
- [Detecting AI-Generated Text In-Depth](#detecting-ai-generated-text-in-depth)
- [Graph Neural Networks (GNN) In-Depth](#graph-neural-networks-gnn-in-depth)
- [High-Performance Data Analysis with Polars](#high-performance-data-analysis-with-polars)
- [Data Quality Validation with Pandera](#data-quality-validation-with-pandera)
- [Local LLMs for Research (No API Needed)](#local-llms-for-research-no-api-needed)
- [Spatial Autocorrelation (Moran's I)](#spatial-autocorrelation-moran's-i)

---

### WhatsApp Chat Analysis In-Depth

**File:** `01_whatsapp_chat_analysis.ipynb`

**Description:** This notebook provides a comprehensive toolkit for analyzing exported WhatsApp chat data. Unlike basic tutorials, this notebook covers: 1.  **Robust Data Generation**: Creating a realistic synthetic dataset for testing. 2.  **Advanced Parsing**: Handling multiple device formats (iOS/Android) and multi-line messages. 3.  **Temporal Analysis**: Activity heatmaps (Day x Hour) and response time distributions. 4.  **Content Analysis**: Sentiment tracking, Emoji frequency, and Word Clouds. 5.  **Interaction Network**: visualizing who replies to whom. - `pandas`: Data manipulation. - `seaborn` / `matplotlib`: Visualization. - `emoji`: Extracting and counting emojis. - `textblob`: Sentiment analysis. - `wordcloud`: Visualizing frequent terms.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/01_whatsapp_chat_analysis.ipynb)

---

### Sentiment Analysis In-Depth

**File:** `02_sentiment_analysis.ipynb`

**Description:** This notebook provides a deep dive into sentiment analysis using two popular libraries: TextBlob and NLTK (VADER). We will compare their performance on a comprehensive synthetic dataset of movie reviews, clean the text, and visualize the results.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/02_sentiment_analysis.ipynb)

---

### Topic Modeling (LDA) In-Depth

**File:** `03_topic_modeling.ipynb`

**Description:** Topic modeling is an unsupervised machine learning technique used to discover abstract "topics" that occur in a collection of documents. This notebook demonstrates how to use **Latent Dirichlet Allocation (LDA)** with Scikit-Learn to extract themes from a synthetic dataset of research abstracts. 1.  Generate a synthetic dataset with distinct underlying themes. 2.  Preprocess text data (Tokenization, Stopword Removal). 3.  Vectorize text using CountVectorizer (Bag of Words). 4.  Fit an LDA model. 5.  Visualize and interpret the topics.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/03_topic_modeling.ipynb)

---

### Word Cloud & N-Grams In-Depth

**File:** `04_wordcloud_ngrams.ipynb`

**Description:** This notebook explores text visualization techniques beyond simple frequency counts. We will generate a larger synthetic dataset, preprocess it, and visualize not just single words (unigrams) but also pairs (bigrams) and triplets (trigrams) to understand context.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/04_wordcloud_ngrams.ipynb)

---

### Named Entity Recognition (NER) In-Depth

**File:** `05_named_entity_recognition.ipynb`

**Description:** NER involves identifying and classifying named entities (people, organizations, locations, dates) in text. This notebook compares NLTK (rule-based/statistical) and SpaCy (model-based) approaches, and visualizes the results on a synthetic news article.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/05_named_entity_recognition.ipynb)

---

### Hate Speech / Toxicity Detection In-Depth

**File:** `06_hate_speech_detection.ipynb`

**Description:** This notebook demonstrates a supervised learning pipeline for detecting toxic comments. We will generate a synthetic dataset, extract features using TF-IDF, train a Logistic Regression model, and evaluating its performance using industry-standard metrics.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/06_hate_speech_detection.ipynb)

---

### Text Preprocessing Pipeline In-Depth

**File:** `07_text_preprocessing.ipynb`

**Description:** Preprocessing is the foundation of NLP. A bad cleaning pipeline creates "garbage in, garbage out". This notebook demonstrates a robust pipeline handling URLs, HTML tags, emojis, and compares Stemming vs. Lemmatization.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/07_text_preprocessing.ipynb)

---

### Survey Data Cleaning: An In-Depth Guide

**File:** `08_survey_cleaning.ipynb`

**Description:** Real-world survey data is rarely clean. It contains duplicates, inconsistent spelling, outliers, and missing values. This notebook demonstrates a professional workflow for cleaning a messy dataset using `pandas`. 1.  **Detect and remove duplicates.** 2.  **Standardize categorical variables** (e.g., mapping "m", "Male", "M" to "Male"). 3.  **Handle missing data** effectively (Drop vs Impute). 4.  **Detect outliers** in numeric responses. 5.  **Recode reverse-coded items** (common in psychological surveys).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/08_survey_cleaning.ipynb)

---

### Descriptive Statistics In-Depth

**File:** `09_descriptive_stats.ipynb`

**Description:** This notebook goes beyond basic summaries to explore robust statistics, distribution shapes, and grouped analyses using a realistic synthetic dataset.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/09_descriptive_stats.ipynb)

---

### Capstone Project: End-to-End Social Media Analysis

**File:** `100_capstone_project.ipynb`

**Description:** This notebook represents a comprehensive workflow, taking you from raw data generation to advanced causal inference. It integrates skills learned throughout the previous notebooks. **Scenario:** You are a social scientist analyzing the impact of a new digital well-being policy introduced on **July 1st, 2023**. *   **Treatment Group**: Users in the "South" region (where the policy was enforced). *   **Control Group**: Users in the "North" region (no policy). *   **Outcome**: Sentiment of social media posts. **Steps:** 1.  **Data Simulation**: Generating realistic text and metadata. 2.  **Preprocessing**: Cleaning text and extracting sentiment. 3.  **EDA**: Visualizing trends over time. 4.  **Causal Inference**: Using Difference-in-Differences (DiD) to estimate the policy's effect.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/100_capstone_project.ipynb)

---

### Likert Scale Analysis In-Depth

**File:** `10_likert_scale_analysis.ipynb`

**Description:** This notebook focuses on analyzing and visualizing survey data collected on a Likert scale (e.g., Strongly Disagree to Strongly Agree). We will cover reliability analysis and specialized visualizations.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/10_likert_scale_analysis.ipynb)

---

### Group Comparisons: T-tests & ANOVA In-Depth

**File:** `11_group_comparisons.ipynb`

**Description:** A fundamental task in social science is comparing means between groups (e.g., Treatment vs Control). This notebook goes beyond simple p-values to include: 1.  **Assumption Checking**: Normality and Homogeneity of Variance. 2.  **Effect Size**: Cohen's d (t-test) and Eta-squared (ANOVA). 3.  **Visualization**: Boxplots with jitter points to show raw data. 4.  **Post-hoc Testing**: Determining exactly *which* groups differ.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/11_group_comparisons.ipynb)

---

### Linear Regression Analysis In-Depth

**File:** `12_linear_regression.ipynb`

**Description:** This notebook covers linear regression with a focus on statistical inference and assumption checking, which is crucial for social science research. We will use `statsmodels` for detailed output.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/12_linear_regression.ipynb)

---

### Logistic Regression In-Depth

**File:** `13_logistic_regression.ipynb`

**Description:** Predicting binary outcomes (Voting Behavior) using Logistic Regression. We will use `statsmodels` for statistical inference (Odds Ratios) and `scikit-learn` for predictive evaluation.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/13_logistic_regression.ipynb)

---

### Exploratory Factor Analysis (EFA) In-Depth

**File:** `14_factor_analysis.ipynb`

**Description:** This notebook demonstrates how to perform Exploratory Factor Analysis (EFA) to uncover latent structures in survey data. We use the `factor_analyzer` library to identify underlying factors (e.g., personality traits) from a set of observed variables (Likert items).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/14_factor_analysis.ipynb)

---

### Cluster Analysis In-Depth

**File:** `15_cluster_analysis.ipynb`

**Description:** Cluster analysis is an unsupervised machine learning technique used to group similar data points together. In social science, this is often used for **segmentation**—identifying distinct subgroups within a population (e.g., voters, consumers, survey respondents) without predefined labels. In this notebook, we will: 1.  Generate a realistic synthetic dataset of **Online Shoppers** (Age, Income, Spending Score). 2.  Preprocess the data (Standardization). 3.  Determine the optimal number of clusters using the **Elbow Method** and **Silhouette Analysis**. 4.  Perform **K-Means Clustering**. 5.  Visualize and interpret the resulting segments.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/15_cluster_analysis.ipynb)

---

### Social Network Construction & Analysis In-Depth

**File:** `16_network_construction.ipynb`

**Description:** This notebook moves beyond basic graph creation to explore complex network analysis techniques using real-world-like data. We will analyze the famous Zachary's Karate Club graph and simulate a larger scale-free network.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/16_network_construction.ipynb)

---

### Network Centrality Measures In-Depth

**File:** `17_network_centrality.ipynb`

**Description:** Centrality measures allow us to identify the most important nodes in a network. In social networks, these might be influencers, bridges between communities, or efficient communicators. In this notebook, we will: 1.  Generate a **Synthetic Social Network** (Scale-Free) using the Barabási-Albert model. 2.  Assign fictitious identities to nodes using `Faker`. 3.  Calculate key centrality measures: *   **Degree Centrality**: Popularity (number of connections). *   **Betweenness Centrality**: Control of information flow (bridges). *   **Closeness Centrality**: Efficiency (average distance to others). *   **Eigenvector Centrality**: Influence (connected to other well-connected nodes). 4.  Visualize the network highlighting influential nodes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/17_network_centrality.ipynb)

---

### Community Detection In-Depth

**File:** `18_community_detection.ipynb`

**Description:** Identifying subgroups within a network using Louvain or Greedy Modularity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/18_community_detection.ipynb)

---

### Time Series Analysis In-Depth

**File:** `19_time_series_analysis.ipynb`

**Description:** Analyzing trends and seasonality in time-series data using classical statistical methods.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/19_time_series_analysis.ipynb)

---

### Geospatial Mapping In-Depth

**File:** `20_geospatial_mapping.ipynb`

**Description:** Visualizing data on maps.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/20_geospatial_mapping.ipynb)

---

### Automated Coding Assistant In-Depth

**File:** `21_auto_coding.ipynb`

**Description:** Keyword-based extraction to assist qualitative coding.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/21_auto_coding.ipynb)

---

### Inter-rater Reliability In-Depth

**File:** `22_inter_rater_reliability.ipynb`

**Description:** Calculating Cohen's Kappa.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/22_inter_rater_reliability.ipynb)

---

### Web Scraping for Social Science: An In-Depth Guide

**File:** `23_web_scraping.ipynb`

**Description:** Web scraping is a powerful tool for collecting digital trace data. This notebook moves beyond simple tutorials to cover robust, ethical, and structured data collection. 1.  **Ethical Scraping**: `robots.txt` and rate limiting. 2.  **HTML Parsing**: Navigating the DOM with `BeautifulSoup`. 3.  **Pagination**: looping through multiple pages. 4.  **Error Handling**: Managing timeouts and missing elements. 5.  **Data Structuring**: Saving results to a clean CSV.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/23_web_scraping.ipynb)

---

### API Data Collection In-Depth

**File:** `24_api_data_collection.ipynb`

**Description:** Fetching data from a public API.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/24_api_data_collection.ipynb)

---

### Data Format Converter In-Depth

**File:** `25_data_converter.ipynb`

**Description:** Converting between JSON, CSV, and Excel formats.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/25_data_converter.ipynb)

---

### Missing Data Imputation In-Depth

**File:** `26_missing_data_imputation.ipynb`

**Description:** Advanced techniques for handling missing values using KNN.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/26_missing_data_imputation.ipynb)

---

### Merge and Join Datasets In-Depth

**File:** `27_merge_datasets.ipynb`

**Description:** Combining data from multiple sources.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/27_merge_datasets.ipynb)

---

### Outlier Detection In-Depth

**File:** `28_outlier_detection.ipynb`

**Description:** Identifying and handling outliers is crucial in social science research. We explore three methods: 1.  **Z-Score**: Parametric method (assumes normality). 2.  **IQR Method**: Non-parametric method (robust to extreme values). 3.  **Isolation Forest**: Machine Learning method for high-dimensional or complex data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/28_outlier_detection.ipynb)

---

### Causal Inference: Propensity Score Matching (PSM) In-Depth

**File:** `29_causal_inference.ipynb`

**Description:** In observational studies, we cannot randomly assign treatment. This leads to **selection bias**: the treated group might be fundamentally different from the control group. **Propensity Score Matching (PSM)** attempts to mimic a randomized control trial (RCT) by matching treated units with control units that have similar probabilities of receiving the treatment (Propensity Scores).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/29_causal_inference.ipynb)

---

### Model Evaluation Metrics In-Depth

**File:** `30_model_evaluation.ipynb`

**Description:** Confusion matrices, ROC curves.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/30_model_evaluation.ipynb)

---

### Multilevel Modeling (Hierarchical Linear Models) In-Depth

**File:** `31_multilevel_modeling.ipynb`

**Description:** Analyzing data where observations are nested within groups (e.g., students within schools).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/31_multilevel_modeling.ipynb)

---

### Survival Analysis In-Depth

**File:** `32_survival_analysis.ipynb`

**Description:** Analyzing time-to-event data (e.g., time until employment, customer churn) using Lifelines.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/32_survival_analysis.ipynb)

---

### Difference-in-Differences (DiD) In-Depth

**File:** `33_difference_in_differences.ipynb`

**Description:** Policy evaluation technique comparing treatment and control groups over time.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/33_difference_in_differences.ipynb)

---

### Regression Discontinuity Design (RDD) In-Depth

**File:** `34_regression_discontinuity.ipynb`

**Description:** Causal inference near a cutoff point.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/34_regression_discontinuity.ipynb)

---

### Fixed Effects Model (Panel Data) In-Depth

**File:** `35_fixed_effects_panel.ipynb`

**Description:** Controlling for time-invariant unobserved heterogeneity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/35_fixed_effects_panel.ipynb)

---

### Mediation Analysis In-Depth

**File:** `36_mediation_analysis.ipynb`

**Description:** Testing if the effect of X on Y is mediated by M (X -> M -> Y).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/36_mediation_analysis.ipynb)

---

### Moderation Analysis In-Depth

**File:** `37_moderation_analysis.ipynb`

**Description:** Testing if the relationship between X and Y depends on Z (Interaction).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/37_moderation_analysis.ipynb)

---

### Meta-Analysis Visualization In-Depth

**File:** `38_meta_analysis.ipynb`

**Description:** Visualizing effect sizes from multiple studies using a Forest Plot.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/38_meta_analysis.ipynb)

---

### Document Similarity In-Depth

**File:** `39_document_similarity.ipynb`

**Description:** Calculating Cosine Similarity between documents.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/39_document_similarity.ipynb)

---

### Supervised Text Classification In-Depth

**File:** `40_text_classification.ipynb`

**Description:** Classifying text into predefined categories using TF-IDF with Linear SVM and Naive Bayes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/40_text_classification.ipynb)

---

### Decision Tree Classifier In-Depth

**File:** `41_decision_tree.ipynb`

**Description:** Interpretable machine learning model.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/41_decision_tree.ipynb)

---

### Random Forest Analysis In-Depth

**File:** `42_random_forest.ipynb`

**Description:** Using Random Forest for classification and feature importance.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/42_random_forest.ipynb)

---

### K-Nearest Neighbors (KNN) In-Depth

**File:** `43_knn_classification.ipynb`

**Description:** Simple classification based on proximity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/43_knn_classification.ipynb)

---

### PDF Text Extraction In-Depth

**File:** `44_pdf_text_extraction.ipynb`

**Description:** Extracting text from PDF files using pypdf. We first create a dummy PDF.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/44_pdf_text_extraction.ipynb)

---

### Nested JSON Flattening In-Depth

**File:** `45_json_flattening.ipynb`

**Description:** Handling complex JSON structures often found in social media API data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/45_json_flattening.ipynb)

---

### Advanced Date and Time Handling In-Depth

**File:** `46_advanced_datetime.ipynb`

**Description:** Handling timezones and calculating durations.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/46_advanced_datetime.ipynb)

---

### Interactive Widgets In-Depth

**File:** `47_interactive_widgets.ipynb`

**Description:** Using ipywidgets for interactive data exploration.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/47_interactive_widgets.ipynb)

---

### Data Anonymization & Privacy In-Depth

**File:** `48_data_anonymization.ipynb`

**Description:** Protecting participant privacy is a core ethical requirement in social science. This notebook demonstrates techniques for de-identifying data, including hashing, k-anonymity, and differential privacy concepts. 1.  **Direct Identifiers**: Names, SSNs, Emails (Must be removed/hashed). 2.  **Quasi-Identifiers (QI)**: Age, Zip Code, Gender (Can be combined to re-identify). 3.  **K-Anonymity**: Ensuring any record is indistinguishable from at least *k-1* other records.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/48_data_anonymization.ipynb)

---

### Multiple Imputation for Missing Data In-Depth

**File:** `49_multiple_imputation.ipynb`

**Description:** Using IterativeImputer (MICE) to handle missing values.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/49_multiple_imputation.ipynb)

---

### Exporting Results In-Depth

**File:** `50_exporting_results.ipynb`

**Description:** Saving DataFrames to nicely formatted Excel files.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/50_exporting_results.ipynb)

---

### Repeated Measures ANOVA In-Depth

**File:** `51_repeated_measures_anova.ipynb`

**Description:** This notebook performs a rigorous Repeated Measures ANOVA using `pingouin`. It includes data generation, assumption checking (Sphericity, Normality), and effect size calculation.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/51_repeated_measures_anova.ipynb)

---

### Mixed ANOVA (Split-Plot Design) In-Depth

**File:** `52_mixed_anova.ipynb`

**Description:** Analyzing designs with both within-subjects and between-subjects factors.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/52_mixed_anova.ipynb)

---

### Signal Detection Theory (SDT) In-Depth

**File:** `53_signal_detection_theory.ipynb`

**Description:** Calculating sensitivity ($d'$) and response bias ($c$ or $\beta$).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/53_signal_detection_theory.ipynb)

---

### Reaction Time Analysis In-Depth

**File:** `54_reaction_time_analysis.ipynb`

**Description:** Preprocessing RT data: Outlier removal, transformations, and visualization.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/54_reaction_time_analysis.ipynb)

---

### Power Analysis In-Depth

**File:** `55_power_analysis.ipynb`

**Description:** Calculating required sample size or achieved power using statsmodels and Pingouin.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/55_power_analysis.ipynb)

---

### Reliability Analysis (Cronbach's Alpha) In-Depth

**File:** `56_reliability_analysis.ipynb`

**Description:** Assessing the internal consistency of a scale/survey.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/56_reliability_analysis.ipynb)

---

### Effect Size Calculation In-Depth

**File:** `57_effect_sizes.ipynb`

**Description:** Calculating Cohen's d, Hedges' g, and Eta-squared.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/57_effect_sizes.ipynb)

---

### Chi-Square Test of Independence In-Depth

**File:** `58_chi_square_test.ipynb`

**Description:** Analyzing the relationship between two categorical variables.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/58_chi_square_test.ipynb)

---

### Mediation Analysis with Bootstrapping In-Depth

**File:** `59_mediation_bootstrapped.ipynb`

**Description:** A more robust method for testing indirect effects using Pingouin.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/59_mediation_bootstrapped.ipynb)

---

### Plotting Interaction Effects In-Depth

**File:** `60_interaction_plots.ipynb`

**Description:** Visualizing 2-way and 3-way interactions using Seaborn.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/60_interaction_plots.ipynb)

---

### Data Analyst Roadmap & Guide In-Depth

**File:** `61_data_analyst_roadmap.ipynb`

**Description:** This notebook serves as a comprehensive guide through the Data Analyst Roadmap, mapping key concepts to practical Python implementations found within this repository. It is aligned with industry standard roadmaps (e.g., [roadmap.sh/data-analyst](https://roadmap.sh/data-analyst)) and covers everything from Excel foundations to Advanced Machine Learning. **Roadmap Sections:** 1.  Introduction & Types of Analytics 2.  Building a Strong Foundation (Excel Equivalents) 3.  Programming Languages (Python vs R) 4.  Data Collection (Databases, APIs, Scraping) 5.  Data Cleanup (Handling Missing Data, Outliers) 6.  Exploratory Data Analysis (EDA) & Visualization 7.  Statistical Analysis 8.  Advanced Topics (Machine Learning, Big Data, Deep Learning) 9.  Projects & Portfolio 10. Online Courses & Certifications 11. Soft Skills & Networking

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/61_data_analyst_roadmap.ipynb)

---

### SQL for Data Analysts In-Depth

**File:** `62_sql_for_analysts.ipynb`

**Description:** Structured Query Language (SQL) is the standard language for managing and manipulating relational databases. For a Data Analyst, it is often considered the most critical technical skill alongside Python/R. In this notebook, we will: 1.  Create a temporary SQLite database in memory. 2.  Populate it with synthetic social science data (Participants and Survey Responses). 3.  Perform essential SQL operations: `SELECT`, `WHERE`, `JOIN`, `GROUP BY`. 4.  Explore advanced concepts: **Window Functions** and **CTEs**. 5.  Integrate SQL with Python using `pandas`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/62_sql_for_analysts.ipynb)

---

### Advanced Visualization with Plotly In-Depth

**File:** `63_advanced_visualization_plotly.ipynb`

**Description:** While `matplotlib` and `seaborn` are excellent for static publications, `plotly` allows for interactive data exploration (zooming, hovering, filtering). In this notebook, we will create: 1.  Interactive Scatter Plots (with hover data) 2.  Interactive Line Charts 3.  Box Plots with points 4.  3D Scatter Plots 5.  Funnel Charts

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/63_advanced_visualization_plotly.ipynb)

---

### Introduction to Transformers for NLP In-Depth

**File:** `64_intro_to_transformers.ipynb`

**Description:** The Transformer architecture (e.g., BERT, GPT) has revolutionized Natural Language Processing (NLP). Unlike older methods (TF-IDF, simple RNNs), Transformers capture deep contextual relationships in text. In this notebook, we will use the **Hugging Face `transformers` library** to perform state-of-the-art NLP tasks with just a few lines of code. **Tasks:** 1.  **Sentiment Analysis**: Determining if a text is positive or negative. 2.  **Zero-Shot Classification**: Classifying text into categories the model hasn't explicitly seen during training.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/64_intro_to_transformers.ipynb)

---

### Bayesian Analysis with PyMC In-Depth

**File:** `65_bayesian_analysis_pymc.ipynb`

**Description:** Bayesian statistics offers a powerful alternative to frequentist (classic) statistics. Instead of calculating a single 'p-value', Bayesian analysis allows us to estimate the **probability distribution** of our parameters given the data. In this notebook, we will use **PyMC** (formerly PyMC3), the standard library for Probabilistic Programming in Python, to perform a simple Bayesian Linear Regression. **Key Concepts:** 1.  **Prior**: What we believe about the parameters *before* seeing data. 2.  **Likelihood**: How likely the data is given the parameters. 3.  **Posterior**: What we believe about the parameters *after* seeing data (Prior + Likelihood).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/65_bayesian_analysis_pymc.ipynb)

---

### Retrieval-Augmented Generation (RAG) for Research In-Depth

**File:** `66_rag_for_research.ipynb`

**Description:** **RAG** is a technique to improve the accuracy of Large Language Models (LLMs) by giving them access to private or specific data. Instead of relying solely on what the model 'knows' from its training, we **retrieve** relevant documents and feed them to the model as context. This is incredibly useful for qualitative researchers who want to "chat" with their interview transcripts or literature review. **Steps:** 1.  **Embed**: Convert text documents into numbers (vectors). 2.  **Index**: Store vectors in a searchable database (FAISS). 3.  **Retrieve**: Find the most relevant documents for a user question. 4.  **Generate**: Ask an LLM to answer the question using the retrieved text.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/66_rag_for_research.ipynb)

---

### Building Interactive Dashboards with Panel In-Depth

**File:** `67_dashboard_with_panel.ipynb`

**Description:** **Panel** is a powerful library that lets you create custom interactive web apps and dashboards by connecting user-defined widgets to plots, images, tables, or text. Unlike Streamlit (which reruns the whole script), Panel is designed to work well within Jupyter Notebooks and can also be deployed as a standalone app. In this notebook, we will build a dashboard to explore a synthetic dataset of **Public Opinion Polling**.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/67_dashboard_with_panel.ipynb)

---

### Causal Inference: Instrumental Variables (IV) In-Depth

**File:** `68_instrumental_variables.ipynb`

**Description:** One of the biggest challenges in social science is **Endogeneity** (Omitted Variable Bias). This happens when there is a hidden variable that affects both your treatment (cause) and your outcome (effect). For example, let's say we want to study the effect of **Education** on **Income**. - Problem: **Ability** (unobserved) affects both Education and Income. - Result: Naive regression overestimates the return to education. **Instrumental Variables (IV)** allow us to isolate the causal effect by using a third variable (Z) that affects Education (X) but has *no direct effect* on Income (Y) except through Education.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/68_instrumental_variables.ipynb)

---

### Python In-Depth for Social Scientists

**File:** `69_python_basics_for_social_scientists.ipynb`

**Description:** This notebook covers the fundamental building blocks of Python programming: Variables, Data Structures (Lists, Dictionaries), Loops, and Functions. We will use social science examples (e.g., ages of participants) to make it practical.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/69_python_basics_for_social_scientists.ipynb)

---

### Pandas Manipulation In-Depth

**File:** `70_pandas_manipulation_basics.ipynb`

**Description:** Pandas is the most powerful tool for tabular data manipulation in Python. Think of it as 'Excel on steroids'. This notebook covers DataFrames, Filtering, and Aggregation.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/70_pandas_manipulation_basics.ipynb)

---

### Data Visualization In-Depth Guide

**File:** `71_data_visualization_101.ipynb`

**Description:** This notebook introduces `matplotlib` and `seaborn` for creating publication-quality plots.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/71_data_visualization_101.ipynb)

---

### Advanced ML: Gradient Boosting with XGBoost In-Depth

**File:** `72_advanced_ml_xgboost.ipynb`

**Description:** XGBoost (eXtreme Gradient Boosting) is one of the most popular and efficient algorithms for structured (tabular) data. It often outperforms other algorithms like Random Forests.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/72_advanced_ml_xgboost.ipynb)

---

### Neural Networks for Tabular Data (PyTorch) In-Depth

**File:** `73_neural_networks_tabular.ipynb`

**Description:** While Deep Learning is famous for Images and NLP, it can also be used for tabular data. This notebook demonstrates how to build a simple Multi-Layer Perceptron (MLP) using PyTorch.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/73_neural_networks_tabular.ipynb)

---

### Exploratory Data Analysis (EDA) In-Depth

**File:** `74_exploratory_data_analysis.ipynb`

**Description:** A comprehensive workflow for inspecting, cleaning, and understanding your data before modeling.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/74_exploratory_data_analysis.ipynb)

---

### MLOps: Model Persistence In-Depth

**File:** `75_mlops_model_persistence.ipynb`

**Description:** Saving and loading Machine Learning models for production use using `joblib` and `pickle`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/75_mlops_model_persistence.ipynb)

---

### Hypothesis Testing Roadmap In-Depth

**File:** `76_hypothesis_testing_roadmap.ipynb`

**Description:** A guide to choosing the correct statistical test for your research question.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/76_hypothesis_testing_roadmap.ipynb)

---

### Audio Transcription with Whisper In-Depth

**File:** `77_audio_transcription.ipynb`

**Description:** This notebook demonstrates how to transcribe audio using OpenAI's Whisper model and generate speech using gTTS.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/77_audio_transcription.ipynb)

---

### Image Analysis In-Depth

**File:** `78_image_analysis_basics.ipynb`

**Description:** This notebook demonstrates how to load, process, and classify images using PyTorch and Torchvision.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/78_image_analysis_basics.ipynb)

---

### Structural Equation Modeling (SEM) In-Depth

**File:** `79_sem_structural_equation_modeling.ipynb`

**Description:** This notebook demonstrates how to perform Structural Equation Modeling using `semopy`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/79_sem_structural_equation_modeling.ipynb)

---

### A/B Testing and Power Analysis In-Depth

**File:** `80_ab_testing_basics.ipynb`

**Description:** This notebook demonstrates how to design and analyze A/B tests using Python.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/80_ab_testing_basics.ipynb)

---

### Conjoint Analysis (Rating-Based) In-Depth

**File:** `81_conjoint_analysis.ipynb`

**Description:** This notebook demonstrates how to perform a Rating-Based Conjoint Analysis. We will generate synthetic data representing consumer preferences for a product (e.g., a Smartphone) with varying attributes (Price, Brand, Storage) and estimate the utility of each attribute level using OLS regression.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/81_conjoint_analysis.ipynb)

---

### Algorithmic Fairness Audit In-Depth

**File:** `82_fairness_audit_in_ai.ipynb`

**Description:** As social scientists, we are often concerned with inequality and bias. When Machine Learning models are used to make decisions (e.g., hiring, loans, policing), they can reproduce or even amplify existing social biases. In this notebook, we will: 1.  Generate a synthetic "Hiring" dataset with intentional gender bias. 2.  Train a standard Logistic Regression model. 3.  Audit the model using `fairlearn` and manual metrics. 4.  Explore mitigation strategies.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/82_fairness_audit_in_ai.ipynb)

---

### Advanced Topic Modeling with BERTopic In-Depth

**File:** `83_advanced_topic_modeling_bertopic.ipynb`

**Description:** Latent Dirichlet Allocation (LDA) (covered in notebook 03) is the classic method for topic modeling. However, it relies on a "Bag-of-Words" approach, ignoring the semantic meaning of words. **BERTopic** is a modern topic modeling technique that leverages **Transformers** (BERT embeddings) to understand the context of words. It works by: 1.  Converting documents to vectors (Embeddings). 2.  Reducing dimensionality (UMAP). 3.  Clustering vectors (HDBSCAN). 4.  Extracting keywords (c-TF-IDF).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/83_advanced_topic_modeling_bertopic.ipynb)

---

### Handling Large Datasets In-Depth

**File:** `84_handling_large_datasets.ipynb`

**Description:** A common hurdle in social science research is the "Out of Memory" error. If you have a 10GB dataset and only 8GB of RAM, you cannot open it with standard `pd.read_csv()`. In this notebook, we explore strategies to handle large data: 1.  **Chunking**: Processing data in small pieces. 2.  **Dask**: Parallel computing with lazy evaluation. 3.  **Efficient Formats**: Using Parquet instead of CSV.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/84_handling_large_datasets.ipynb)

---

### Multimodal Analysis: Text and Image In-Depth

**File:** `85_multimodal_analysis.ipynb`

**Description:** This notebook demonstrates **Multimodal Analysis** using the **CLIP (Contrastive Language-Image Pre-Training)** model via the Hugging Face `transformers` library. Multimodal analysis involves processing and correlating data from different modalities, such as text and images. This is increasingly relevant in social science for analyzing social media content (memes, posts with images). We will cover: 1.  **Zero-Shot Image Classification**: Classifying an image into categories without training a specific model for those categories. 2.  **Image-Text Matching**: Finding the best caption for an image from a list of options. 3.  **Synthetic Image Generation**: Creating simple images with text to test the model's capabilities.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/85_multimodal_analysis.ipynb)

---

### Recommender Systems In-Depth

**File:** `86_recommender_systems_basics.ipynb`

**Description:** Recommender systems are algorithms designed to suggest relevant items to users (e.g., movies to watch, books to read). They are fundamental to modern web applications. In this notebook, we will explore two primary techniques using a synthetic dataset of **Books** and **User Ratings**: 1.  **Content-Based Filtering**: Recommends items similar to those a user liked, based on item attributes (e.g., Book Descriptions). 2.  **Collaborative Filtering**: Recommends items based on the past interactions of similar users (e.g., Matrix Factorization).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/86_recommender_systems_basics.ipynb)

---

### Explainable AI (XAI) with SHAP In-Depth

**File:** `87_explainable_ai_shap.ipynb`

**Description:** This notebook explores **Explainable AI (XAI)** using the **SHAP (SHapley Additive exPlanations)** library. Machine learning models, especially complex ones like Random Forests or Neural Networks, are often "black boxes." SHAP values help us understand *why* a model made a specific prediction by quantifying the contribution of each feature. We will cover: 1.  **Synthetic Data Generation**: Creating a realistic "Credit Risk" dataset. 2.  **Model Training**: Training a Random Forest Classifier to predict default risk. 3.  **SHAP Explainer**: calculating SHAP values. 4.  **Global Interpretability**: Summary plots to see overall feature importance. 5.  **Local Interpretability**: Force plots to explain individual predictions. 6.  **Dependence Plots**: Understanding interaction effects between features.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/87_explainable_ai_shap.ipynb)

---

### Time Series Forecasting with Prophet In-Depth

**File:** `88_time_series_forecasting_prophet.ipynb`

**Description:** This notebook demonstrates time series forecasting using **Facebook Prophet**. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. We will cover: 1.  **Synthetic Data Generation**: Creating a time series with trend and seasonality (e.g., website traffic). 2.  **Model Fitting**: Training the Prophet model. 3.  **Forecasting**: Predicting future values. 4.  **Component Analysis**: Decomposing the forecast into trend and seasonal components.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/88_time_series_forecasting_prophet.ipynb)

---

### Qualitative Analysis with LLMs (Zero-Shot Classification) In-Depth

**File:** `89_qualitative_analysis_with_llms.ipynb`

**Description:** Traditional qualitative coding is time-consuming. Large Language Models (LLMs) can automate the initial coding pass using Zero-Shot Classification. This notebook demonstrates how to categorize synthetic interview transcripts into themes without training a model. **Techniques:** 1.  **Synthetic Interview Generation**: Using `Faker` to simulate qualitative data. 2.  **Zero-Shot Classification**: Using Hugging Face `transformers` to assign themes. 3.  **Visualization**: Comparing predicted themes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/89_qualitative_analysis_with_llms.ipynb)

---

### Agentic AI Concepts: The ReAct Pattern In-Depth

**File:** `90_agentic_ai_concepts.ipynb`

**Description:** Agentic AI goes beyond simple classification or prediction. Agents can *reason* about a problem, *act* by using tools, and *observe* the results to adjust their plan. This pattern is known as **ReAct (Reasoning + Acting)**. In this notebook, we build a **conceptual Agent** from scratch. Instead of using a paid API (like OpenAI), we use a "Mock LLM"—a simple function that simulates how a language model would respond to prompts—to demonstrate the architecture.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/90_agentic_ai_concepts.ipynb)

---

### Handling Imbalanced Datasets (SMOTE & ADASYN) In-Depth

**File:** `91_imbalanced_learning.ipynb`

**Description:** In social science, interesting events are often rare: protests, fraud, hate speech, or disease outbreaks. Standard machine learning models bias towards the majority class, often ignoring the minority class entirely. This notebook demonstrates how to solve this using **Resampling Techniques** from the `imbalanced-learn` library.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/91_imbalanced_learning.ipynb)

---

### Bipartite Network Analysis (Affiliation Networks) In-Depth

**File:** `92_bipartite_network_analysis.ipynb`

**Description:** Most social networks we study are "One-Mode" (Person-Person). However, data often comes in "Two-Mode" or **Bipartite** forms: People attending Events, Directors on Boards, or Authors on Papers. This notebook demonstrates how to handle these networks using `networkx`, specifically projecting them into one-mode networks to study the relationships implied by shared affiliations.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/92_bipartite_network_analysis.ipynb)

---

### Synthetic Control Methods (SCM) In-Depth

**File:** `93_synthetic_control_methods.ipynb`

**Description:** The **Synthetic Control Method (SCM)** is a statistical method used to evaluate the effect of an intervention in comparative case studies. It involves constructing a weighted combination of control units (the "synthetic control") that best resembles the characteristics of the treated unit prior to the intervention. Imagine a city ("City A") implements a new tax policy in year 2010. We want to know the effect of this tax on **GDP per capita**. We have data for City A and 10 other cities that did *not* implement the tax. Instead of just comparing City A to the average of the other cities, we create a "Synthetic City A" which is a weighted average of the other cities, chosen to match City A's GDP trajectory *before* 2010.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/93_synthetic_control_methods.ipynb)

---

### Detecting AI-Generated Text In-Depth

**File:** `94_detecting_ai_generated_text.ipynb`

**Description:** As Large Language Models (LLMs) become ubiquitous, distinguishing between human and machine-generated text is a critical challenge in social science research. Common hypotheses: 1.  **Perplexity (PPL)**: AI models aim to maximize probability, so they often choose "safe", predictable words. This leads to *lower* perplexity than human text. 2.  **Burstiness**: Humans write with varying sentence structures (short, then long and complex). AI tends to be more monotonic. Burstiness measures the variation in perplexity. In this notebook, we will use a small model (**GPT-2**) to calculate these metrics on sample texts.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/94_detecting_ai_generated_text.ipynb)

---

### Graph Neural Networks (GNN) In-Depth

**File:** `95_graph_neural_networks.ipynb`

**Description:** This notebook demonstrates how to build a Graph Convolutional Network (GCN) from scratch using PyTorch to perform node classification on a synthetic Academic Citation Network.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/95_graph_neural_networks.ipynb)

---

### High-Performance Data Analysis with Polars

**File:** `96_polars_data_analysis.ipynb`

**Description:** Polars is a lightning-fast DataFrame library implemented in Rust and memory-efficient. This notebook demonstrates its syntax and performance advantages over Pandas for large datasets.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/96_polars_data_analysis.ipynb)

---

### Data Quality Validation with Pandera

**File:** `97_data_quality_validation.ipynb`

**Description:** In social science research, data quality is paramount. Manual checks are prone to error. `pandera` allows you to define a **schema** for your data and automatically validate it, catching errors early. This notebook covers: 1.  **Generating 'Dirty' Data**: Simulating real-world errors (typos, out-of-range values). 2.  **Defining a Schema**: specifying expected data types and constraints. 3.  **Validation**: Running checks and handling failures. 4.  **Cleaning & Re-validating**: Fixing the data to pass the schema.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/97_data_quality_validation.ipynb)

---

### Local LLMs for Research (No API Needed)

**File:** `98_local_llms_for_research.ipynb`

**Description:** Using proprietary APIs (like OpenAI) can be costly and raise privacy concerns. This notebook demonstrates how to run **local** Large Language Models (LLMs) on your own machine (or Colab) using the Hugging Face `transformers` library. We use `google/flan-t5-small`, a lightweight model capable of summarization, translation, and Q&A, which runs efficiently even on a CPU.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/98_local_llms_for_research.ipynb)

---

### Spatial Autocorrelation (Moran's I)

**File:** `99_spatial_autocorrelation.ipynb`

**Description:** Spatial autocorrelation measures how correlated a variable is with itself in space. -   **Positive**: High values cluster near high values (Hotspots). -   **Negative**: High values are near low values (Checkerboard). -   **Zero**: Random distribution. We will use **Moran's I** to quantify this on a synthetic grid.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/99_spatial_autocorrelation.ipynb)

---
