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
- [Likert Scale Analysis In-Depth](#likert-scale-analysis-in-depth)
- [Group Comparisons: T-tests & ANOVA](#group-comparisons:-t-tests-anova)
- [Linear Regression Analysis In-Depth](#linear-regression-analysis-in-depth)
- [Logistic Regression In-Depth](#logistic-regression-in-depth)
- [Exploratory Factor Analysis (EFA)](#exploratory-factor-analysis-efa)
- [Cluster Analysis](#cluster-analysis)
- [Social Network Construction & Analysis](#social-network-construction-analysis)
- [Network Centrality Measures](#network-centrality-measures)
- [Community Detection](#community-detection)
- [Time Series Analysis](#time-series-analysis)
- [Geospatial Mapping](#geospatial-mapping)
- [Automated Coding Assistant](#automated-coding-assistant)
- [Inter-rater Reliability](#inter-rater-reliability)
- [Web Scraping for Social Science: An In-Depth Guide](#web-scraping-for-social-science:-an-in-depth-guide)
- [API Data Collection](#api-data-collection)
- [Data Format Converter](#data-format-converter)
- [Missing Data Imputation](#missing-data-imputation)
- [Merge and Join Datasets](#merge-and-join-datasets)
- [Outlier Detection In-Depth](#outlier-detection-in-depth)
- [Causal Inference: Propensity Score Matching (PSM)](#causal-inference:-propensity-score-matching-psm)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Multilevel Modeling (Hierarchical Linear Models)](#multilevel-modeling-hierarchical-linear-models)
- [Survival Analysis](#survival-analysis)
- [Difference-in-Differences (DiD)](#difference-in-differences-did)
- [Regression Discontinuity Design (RDD)](#regression-discontinuity-design-rdd)
- [Fixed Effects Model (Panel Data)](#fixed-effects-model-panel-data)
- [Mediation Analysis](#mediation-analysis)
- [Moderation Analysis](#moderation-analysis)
- [Meta-Analysis Visualization](#meta-analysis-visualization)
- [Document Similarity](#document-similarity)
- [Supervised Text Classification](#supervised-text-classification)
- [Decision Tree Classifier](#decision-tree-classifier)
- [Random Forest Analysis](#random-forest-analysis)
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [PDF Text Extraction](#pdf-text-extraction)
- [Nested JSON Flattening](#nested-json-flattening)
- [Advanced Date and Time Handling](#advanced-date-and-time-handling)
- [Interactive Widgets](#interactive-widgets)
- [Data Anonymization & Privacy](#data-anonymization-privacy)
- [Multiple Imputation for Missing Data](#multiple-imputation-for-missing-data)
- [Exporting Results](#exporting-results)
- [Repeated Measures ANOVA In-Depth](#repeated-measures-anova-in-depth)
- [Mixed ANOVA (Split-Plot Design)](#mixed-anova-split-plot-design)
- [Signal Detection Theory (SDT)](#signal-detection-theory-sdt)
- [Reaction Time Analysis](#reaction-time-analysis)
- [Power Analysis](#power-analysis)
- [Reliability Analysis (Cronbach's Alpha)](#reliability-analysis-cronbach's-alpha)
- [Effect Size Calculation](#effect-size-calculation)
- [Chi-Square Test of Independence](#chi-square-test-of-independence)
- [Mediation Analysis with Bootstrapping](#mediation-analysis-with-bootstrapping)
- [Plotting Interaction Effects](#plotting-interaction-effects)
- [Data Analyst Roadmap & Guide](#data-analyst-roadmap-guide)
- [SQL for Data Analysts](#sql-for-data-analysts)
- [Advanced Visualization with Plotly](#advanced-visualization-with-plotly)
- [Introduction to Transformers for NLP](#introduction-to-transformers-for-nlp)
- [Bayesian Analysis with PyMC](#bayesian-analysis-with-pymc)
- [Retrieval-Augmented Generation (RAG) for Research](#retrieval-augmented-generation-rag-for-research)
- [Building Interactive Dashboards with Panel](#building-interactive-dashboards-with-panel)
- [Causal Inference: Instrumental Variables (IV)](#causal-inference:-instrumental-variables-iv)
- [Python Basics for Social Scientists](#python-basics-for-social-scientists)
- [Pandas Manipulation Basics](#pandas-manipulation-basics)
- [Data Visualization 101](#data-visualization-101)
- [Advanced ML: Gradient Boosting with XGBoost](#advanced-ml:-gradient-boosting-with-xgboost)
- [Neural Networks for Tabular Data (PyTorch)](#neural-networks-for-tabular-data-pytorch)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [MLOps: Model Persistence](#mlops:-model-persistence)
- [Hypothesis Testing Roadmap](#hypothesis-testing-roadmap)
- [Audio Transcription with Whisper](#audio-transcription-with-whisper)
- [Image Analysis Basics](#image-analysis-basics)
- [Structural Equation Modeling (SEM)](#structural-equation-modeling-sem)
- [A/B Testing and Power Analysis](#ab-testing-and-power-analysis)
- [Conjoint Analysis (Rating-Based)](#conjoint-analysis-rating-based)
- [Algorithmic Fairness Audit](#algorithmic-fairness-audit)
- [Advanced Topic Modeling with BERTopic](#advanced-topic-modeling-with-bertopic)
- [Handling Large Datasets](#handling-large-datasets)

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

### Likert Scale Analysis In-Depth

**File:** `10_likert_scale_analysis.ipynb`

**Description:** This notebook focuses on analyzing and visualizing survey data collected on a Likert scale (e.g., Strongly Disagree to Strongly Agree). We will cover reliability analysis and specialized visualizations.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/10_likert_scale_analysis.ipynb)

---

### Group Comparisons: T-tests & ANOVA

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

### Exploratory Factor Analysis (EFA)

**File:** `14_factor_analysis.ipynb`

**Description:** This notebook demonstrates how to perform Exploratory Factor Analysis (EFA) to uncover latent structures in survey data. We use the `factor_analyzer` library to identify underlying factors (e.g., personality traits) from a set of observed variables (Likert items).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/14_factor_analysis.ipynb)

---

### Cluster Analysis

**File:** `15_cluster_analysis.ipynb`

**Description:** Segmenting data using K-Means.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/15_cluster_analysis.ipynb)

---

### Social Network Construction & Analysis

**File:** `16_network_construction.ipynb`

**Description:** This notebook moves beyond basic graph creation to explore complex network analysis techniques using real-world-like data. We will analyze the famous Zachary's Karate Club graph and simulate a larger scale-free network.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/16_network_construction.ipynb)

---

### Network Centrality Measures

**File:** `17_network_centrality.ipynb`

**Description:** Calculating Degree, Betweenness, and Closeness centrality.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/17_network_centrality.ipynb)

---

### Community Detection

**File:** `18_community_detection.ipynb`

**Description:** Identifying subgroups within a network using Louvain or Greedy Modularity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/18_community_detection.ipynb)

---

### Time Series Analysis

**File:** `19_time_series_analysis.ipynb`

**Description:** Analyzing trends and seasonality in time-series data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/19_time_series_analysis.ipynb)

---

### Geospatial Mapping

**File:** `20_geospatial_mapping.ipynb`

**Description:** Visualizing data on maps.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/20_geospatial_mapping.ipynb)

---

### Automated Coding Assistant

**File:** `21_auto_coding.ipynb`

**Description:** Keyword-based extraction to assist qualitative coding.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/21_auto_coding.ipynb)

---

### Inter-rater Reliability

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

### API Data Collection

**File:** `24_api_data_collection.ipynb`

**Description:** Fetching data from a public API.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/24_api_data_collection.ipynb)

---

### Data Format Converter

**File:** `25_data_converter.ipynb`

**Description:** Converting between JSON, CSV, and Excel formats.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/25_data_converter.ipynb)

---

### Missing Data Imputation

**File:** `26_missing_data_imputation.ipynb`

**Description:** Advanced techniques for handling missing values using KNN.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/26_missing_data_imputation.ipynb)

---

### Merge and Join Datasets

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

### Causal Inference: Propensity Score Matching (PSM)

**File:** `29_causal_inference.ipynb`

**Description:** In observational studies, we cannot randomly assign treatment. This leads to **selection bias**: the treated group might be fundamentally different from the control group. **Propensity Score Matching (PSM)** attempts to mimic a randomized control trial (RCT) by matching treated units with control units that have similar probabilities of receiving the treatment (Propensity Scores).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/29_causal_inference.ipynb)

---

### Model Evaluation Metrics

**File:** `30_model_evaluation.ipynb`

**Description:** Confusion matrices, ROC curves.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/30_model_evaluation.ipynb)

---

### Multilevel Modeling (Hierarchical Linear Models)

**File:** `31_multilevel_modeling.ipynb`

**Description:** Analyzing data where observations are nested within groups (e.g., students within schools).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/31_multilevel_modeling.ipynb)

---

### Survival Analysis

**File:** `32_survival_analysis.ipynb`

**Description:** Analyzing time-to-event data (e.g., time until employment, customer churn) using Lifelines.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/32_survival_analysis.ipynb)

---

### Difference-in-Differences (DiD)

**File:** `33_difference_in_differences.ipynb`

**Description:** Policy evaluation technique comparing treatment and control groups over time.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/33_difference_in_differences.ipynb)

---

### Regression Discontinuity Design (RDD)

**File:** `34_regression_discontinuity.ipynb`

**Description:** Causal inference near a cutoff point.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/34_regression_discontinuity.ipynb)

---

### Fixed Effects Model (Panel Data)

**File:** `35_fixed_effects_panel.ipynb`

**Description:** Controlling for time-invariant unobserved heterogeneity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/35_fixed_effects_panel.ipynb)

---

### Mediation Analysis

**File:** `36_mediation_analysis.ipynb`

**Description:** Testing if the effect of X on Y is mediated by M (X -> M -> Y).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/36_mediation_analysis.ipynb)

---

### Moderation Analysis

**File:** `37_moderation_analysis.ipynb`

**Description:** Testing if the relationship between X and Y depends on Z (Interaction).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/37_moderation_analysis.ipynb)

---

### Meta-Analysis Visualization

**File:** `38_meta_analysis.ipynb`

**Description:** Visualizing effect sizes from multiple studies using a Forest Plot.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/38_meta_analysis.ipynb)

---

### Document Similarity

**File:** `39_document_similarity.ipynb`

**Description:** Calculating Cosine Similarity between documents.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/39_document_similarity.ipynb)

---

### Supervised Text Classification

**File:** `40_text_classification.ipynb`

**Description:** Classifying text into predefined categories using TF-IDF with Linear SVM and Naive Bayes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/40_text_classification.ipynb)

---

### Decision Tree Classifier

**File:** `41_decision_tree.ipynb`

**Description:** Interpretable machine learning model.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/41_decision_tree.ipynb)

---

### Random Forest Analysis

**File:** `42_random_forest.ipynb`

**Description:** Using Random Forest for classification and feature importance.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/42_random_forest.ipynb)

---

### K-Nearest Neighbors (KNN)

**File:** `43_knn_classification.ipynb`

**Description:** Simple classification based on proximity.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/43_knn_classification.ipynb)

---

### PDF Text Extraction

**File:** `44_pdf_text_extraction.ipynb`

**Description:** Extracting text from PDF files using pypdf. We first create a dummy PDF.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/44_pdf_text_extraction.ipynb)

---

### Nested JSON Flattening

**File:** `45_json_flattening.ipynb`

**Description:** Handling complex JSON structures often found in social media API data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/45_json_flattening.ipynb)

---

### Advanced Date and Time Handling

**File:** `46_advanced_datetime.ipynb`

**Description:** Handling timezones and calculating durations.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/46_advanced_datetime.ipynb)

---

### Interactive Widgets

**File:** `47_interactive_widgets.ipynb`

**Description:** Using ipywidgets for interactive data exploration.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/47_interactive_widgets.ipynb)

---

### Data Anonymization & Privacy

**File:** `48_data_anonymization.ipynb`

**Description:** Protecting participant privacy is a core ethical requirement in social science. This notebook demonstrates techniques for de-identifying data, including hashing, k-anonymity, and differential privacy concepts. 1.  **Direct Identifiers**: Names, SSNs, Emails (Must be removed/hashed). 2.  **Quasi-Identifiers (QI)**: Age, Zip Code, Gender (Can be combined to re-identify). 3.  **K-Anonymity**: Ensuring any record is indistinguishable from at least *k-1* other records.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/48_data_anonymization.ipynb)

---

### Multiple Imputation for Missing Data

**File:** `49_multiple_imputation.ipynb`

**Description:** Using IterativeImputer (MICE) to handle missing values.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/49_multiple_imputation.ipynb)

---

### Exporting Results

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

### Mixed ANOVA (Split-Plot Design)

**File:** `52_mixed_anova.ipynb`

**Description:** Analyzing designs with both within-subjects and between-subjects factors.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/52_mixed_anova.ipynb)

---

### Signal Detection Theory (SDT)

**File:** `53_signal_detection_theory.ipynb`

**Description:** Calculating sensitivity ($d'$) and response bias ($c$ or $\beta$).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/53_signal_detection_theory.ipynb)

---

### Reaction Time Analysis

**File:** `54_reaction_time_analysis.ipynb`

**Description:** Preprocessing RT data: Outlier removal, transformations, and visualization.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/54_reaction_time_analysis.ipynb)

---

### Power Analysis

**File:** `55_power_analysis.ipynb`

**Description:** Calculating required sample size or achieved power using statsmodels and Pingouin.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/55_power_analysis.ipynb)

---

### Reliability Analysis (Cronbach's Alpha)

**File:** `56_reliability_analysis.ipynb`

**Description:** Assessing the internal consistency of a scale/survey.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/56_reliability_analysis.ipynb)

---

### Effect Size Calculation

**File:** `57_effect_sizes.ipynb`

**Description:** Calculating Cohen's d, Hedges' g, and Eta-squared.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/57_effect_sizes.ipynb)

---

### Chi-Square Test of Independence

**File:** `58_chi_square_test.ipynb`

**Description:** Analyzing the relationship between two categorical variables.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/58_chi_square_test.ipynb)

---

### Mediation Analysis with Bootstrapping

**File:** `59_mediation_bootstrapped.ipynb`

**Description:** A more robust method for testing indirect effects using Pingouin.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/59_mediation_bootstrapped.ipynb)

---

### Plotting Interaction Effects

**File:** `60_interaction_plots.ipynb`

**Description:** Visualizing 2-way and 3-way interactions using Seaborn.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/60_interaction_plots.ipynb)

---

### Data Analyst Roadmap & Guide

**File:** `61_data_analyst_roadmap.ipynb`

**Description:** This notebook serves as a comprehensive guide through the Data Analyst Roadmap, mapping key concepts to practical Python implementations found within this repository. It is aligned with industry standard roadmaps (e.g., [roadmap.sh/data-analyst](https://roadmap.sh/data-analyst)) and covers everything from Excel foundations to Advanced Machine Learning. **Roadmap Sections:** 1.  Introduction & Types of Analytics 2.  Building a Strong Foundation (Excel Equivalents) 3.  Programming Languages (Python) 4.  Data Collection (Databases, APIs, Scraping) 5.  Data Cleanup (Handling Missing Data, Outliers) 6.  Exploratory Data Analysis (EDA) & Visualization 7.  Statistical Analysis 8.  Advanced Topics (Machine Learning, Deep Learning, Big Data) 9.  Projects & Portfolio 10. Soft Skills & Networking

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/61_data_analyst_roadmap.ipynb)

---

### SQL for Data Analysts

**File:** `62_sql_for_analysts.ipynb`

**Description:** Structured Query Language (SQL) is the standard language for managing and manipulating relational databases. For a Data Analyst, it is often considered the most critical technical skill alongside Python/R. In this notebook, we will: 1.  Create a temporary SQLite database in memory. 2.  Populate it with synthetic social science data (Participants and Survey Responses). 3.  Perform essential SQL operations: `SELECT`, `WHERE`, `JOIN`, `GROUP BY`. 4.  Integrate SQL with Python using `pandas`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/62_sql_for_analysts.ipynb)

---

### Advanced Visualization with Plotly

**File:** `63_advanced_visualization_plotly.ipynb`

**Description:** While `matplotlib` and `seaborn` are excellent for static publications, `plotly` allows for interactive data exploration (zooming, hovering, filtering). In this notebook, we will create: 1.  Interactive Scatter Plots (with hover data) 2.  Interactive Line Charts 3.  Box Plots with points 4.  3D Scatter Plots 5.  Funnel Charts

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/63_advanced_visualization_plotly.ipynb)

---

### Introduction to Transformers for NLP

**File:** `64_intro_to_transformers.ipynb`

**Description:** The Transformer architecture (e.g., BERT, GPT) has revolutionized Natural Language Processing (NLP). Unlike older methods (TF-IDF, simple RNNs), Transformers capture deep contextual relationships in text. In this notebook, we will use the **Hugging Face `transformers` library** to perform state-of-the-art NLP tasks with just a few lines of code. **Tasks:** 1.  **Sentiment Analysis**: Determining if a text is positive or negative. 2.  **Zero-Shot Classification**: Classifying text into categories the model hasn't explicitly seen during training.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/64_intro_to_transformers.ipynb)

---

### Bayesian Analysis with PyMC

**File:** `65_bayesian_analysis_pymc.ipynb`

**Description:** Bayesian statistics offers a powerful alternative to frequentist (classic) statistics. Instead of calculating a single 'p-value', Bayesian analysis allows us to estimate the **probability distribution** of our parameters given the data. In this notebook, we will use **PyMC** (formerly PyMC3), the standard library for Probabilistic Programming in Python, to perform a simple Bayesian Linear Regression. **Key Concepts:** 1.  **Prior**: What we believe about the parameters *before* seeing data. 2.  **Likelihood**: How likely the data is given the parameters. 3.  **Posterior**: What we believe about the parameters *after* seeing data (Prior + Likelihood).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/65_bayesian_analysis_pymc.ipynb)

---

### Retrieval-Augmented Generation (RAG) for Research

**File:** `66_rag_for_research.ipynb`

**Description:** **RAG** is a technique to improve the accuracy of Large Language Models (LLMs) by giving them access to private or specific data. Instead of relying solely on what the model 'knows' from its training, we **retrieve** relevant documents and feed them to the model as context. This is incredibly useful for qualitative researchers who want to "chat" with their interview transcripts or literature review. **Steps:** 1.  **Embed**: Convert text documents into numbers (vectors). 2.  **Index**: Store vectors in a searchable database (FAISS). 3.  **Retrieve**: Find the most relevant documents for a user question. 4.  **Generate**: Ask an LLM to answer the question using the retrieved text.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/66_rag_for_research.ipynb)

---

### Building Interactive Dashboards with Panel

**File:** `67_dashboard_with_panel.ipynb`

**Description:** **Panel** is a powerful library that lets you create custom interactive web apps and dashboards by connecting user-defined widgets to plots, images, tables, or text. Unlike Streamlit (which reruns the whole script), Panel is designed to work well within Jupyter Notebooks and can also be deployed as a standalone app. In this notebook, we will build a dashboard to explore a synthetic dataset of **Public Opinion Polling**.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/67_dashboard_with_panel.ipynb)

---

### Causal Inference: Instrumental Variables (IV)

**File:** `68_instrumental_variables.ipynb`

**Description:** One of the biggest challenges in social science is **Endogeneity** (Omitted Variable Bias). This happens when there is a hidden variable that affects both your treatment (cause) and your outcome (effect). For example, let's say we want to study the effect of **Education** on **Income**. - Problem: **Ability** (unobserved) affects both Education and Income. - Result: Naive regression overestimates the return to education. **Instrumental Variables (IV)** allow us to isolate the causal effect by using a third variable (Z) that affects Education (X) but has *no direct effect* on Income (Y) except through Education.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/68_instrumental_variables.ipynb)

---

### Python Basics for Social Scientists

**File:** `69_python_basics_for_social_scientists.ipynb`

**Description:** This notebook covers the fundamental building blocks of Python programming: Variables, Data Structures (Lists, Dictionaries), Loops, and Functions. We will use social science examples (e.g., ages of participants) to make it practical.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/69_python_basics_for_social_scientists.ipynb)

---

### Pandas Manipulation Basics

**File:** `70_pandas_manipulation_basics.ipynb`

**Description:** Pandas is the most powerful tool for tabular data manipulation in Python. Think of it as 'Excel on steroids'. This notebook covers DataFrames, Filtering, and Aggregation.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/70_pandas_manipulation_basics.ipynb)

---

### Data Visualization 101

**File:** `71_data_visualization_101.ipynb`

**Description:** This notebook introduces `matplotlib` and `seaborn` for creating publication-quality plots.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/71_data_visualization_101.ipynb)

---

### Advanced ML: Gradient Boosting with XGBoost

**File:** `72_advanced_ml_xgboost.ipynb`

**Description:** XGBoost (eXtreme Gradient Boosting) is one of the most popular and efficient algorithms for structured (tabular) data. It often outperforms other algorithms like Random Forests.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/72_advanced_ml_xgboost.ipynb)

---

### Neural Networks for Tabular Data (PyTorch)

**File:** `73_neural_networks_tabular.ipynb`

**Description:** While Deep Learning is famous for Images and NLP, it can also be used for tabular data. This notebook demonstrates how to build a simple Multi-Layer Perceptron (MLP) using PyTorch.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/73_neural_networks_tabular.ipynb)

---

### Exploratory Data Analysis (EDA)

**File:** `74_exploratory_data_analysis.ipynb`

**Description:** A comprehensive workflow for inspecting, cleaning, and understanding your data before modeling.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/74_exploratory_data_analysis.ipynb)

---

### MLOps: Model Persistence

**File:** `75_mlops_model_persistence.ipynb`

**Description:** Saving and loading Machine Learning models for production use using `joblib` and `pickle`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/75_mlops_model_persistence.ipynb)

---

### Hypothesis Testing Roadmap

**File:** `76_hypothesis_testing_roadmap.ipynb`

**Description:** A guide to choosing the correct statistical test for your research question.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/76_hypothesis_testing_roadmap.ipynb)

---

### Audio Transcription with Whisper

**File:** `77_audio_transcription.ipynb`

**Description:** This notebook demonstrates how to transcribe audio using OpenAI's Whisper model and generate speech using gTTS.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/77_audio_transcription.ipynb)

---

### Image Analysis Basics

**File:** `78_image_analysis_basics.ipynb`

**Description:** This notebook demonstrates how to load, process, and classify images using PyTorch and Torchvision.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/78_image_analysis_basics.ipynb)

---

### Structural Equation Modeling (SEM)

**File:** `79_sem_structural_equation_modeling.ipynb`

**Description:** This notebook demonstrates how to perform Structural Equation Modeling using `semopy`.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/79_sem_structural_equation_modeling.ipynb)

---

### A/B Testing and Power Analysis

**File:** `80_ab_testing_basics.ipynb`

**Description:** This notebook demonstrates how to design and analyze A/B tests using Python.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/80_ab_testing_basics.ipynb)

---

### Conjoint Analysis (Rating-Based)

**File:** `81_conjoint_analysis.ipynb`

**Description:** This notebook demonstrates how to perform a Rating-Based Conjoint Analysis. We will generate synthetic data representing consumer preferences for a product (e.g., a Smartphone) with varying attributes (Price, Brand, Storage) and estimate the utility of each attribute level using OLS regression.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/81_conjoint_analysis.ipynb)

---

### Algorithmic Fairness Audit

**File:** `82_fairness_audit_in_ai.ipynb`

**Description:** As social scientists, we are often concerned with inequality and bias. When Machine Learning models are used to make decisions (e.g., hiring, loans, policing), they can reproduce or even amplify existing social biases. In this notebook, we will: 1.  Generate a synthetic "Hiring" dataset with intentional gender bias. 2.  Train a standard Logistic Regression model. 3.  Audit the model using `fairlearn` and manual metrics. 4.  Explore mitigation strategies.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/82_fairness_audit_in_ai.ipynb)

---

### Advanced Topic Modeling with BERTopic

**File:** `83_advanced_topic_modeling_bertopic.ipynb`

**Description:** Latent Dirichlet Allocation (LDA) (covered in notebook 03) is the classic method for topic modeling. However, it relies on a "Bag-of-Words" approach, ignoring the semantic meaning of words. **BERTopic** is a modern topic modeling technique that leverages **Transformers** (BERT embeddings) to understand the context of words. It works by: 1.  Converting documents to vectors (Embeddings). 2.  Reducing dimensionality (UMAP). 3.  Clustering vectors (HDBSCAN). 4.  Extracting keywords (c-TF-IDF).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/83_advanced_topic_modeling_bertopic.ipynb)

---

### Handling Large Datasets

**File:** `84_handling_large_datasets.ipynb`

**Description:** A common hurdle in social science research is the "Out of Memory" error. If you have a 10GB dataset and only 8GB of RAM, you cannot open it with standard `pd.read_csv()`. In this notebook, we explore strategies to handle large data: 1.  **Chunking**: Processing data in small pieces. 2.  **Dask**: Parallel computing with lazy evaluation. 3.  **Efficient Formats**: Using Parquet instead of CSV.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/84_handling_large_datasets.ipynb)

---

### Multimodal Analysis: Text and Image

**File:** `85_multimodal_analysis.ipynb`

**Description:** This notebook introduces **Multimodal Analysis** using the **CLIP** model via Hugging Face `transformers`. It covers: 1. **Zero-Shot Image Classification**: Classifying images without specific training. 2. **Image-Text Matching**: correlating text and images. 3. **Synthetic Image Generation**: Creating simple images for testing purposes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/85_multimodal_analysis.ipynb)

---
