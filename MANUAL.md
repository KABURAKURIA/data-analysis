# Social Science Research Notebooks Manual

This manual provides an overview of the tools available in this repository. All tools have been verified as functional.

## Table of Contents

- [WhatsApp Chat Analysis](#whatsapp-chat-analysis)
- [Sentiment Analysis](#sentiment-analysis)
- [Topic Modeling (LDA)](#topic-modeling-lda)
- [Word Cloud & N-Grams](#word-cloud-n-grams)
- [Named Entity Recognition (NER)](#named-entity-recognition-ner)
- [Hate Speech/Toxicity Detection](#hate-speechtoxicity-detection)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Survey Data Cleaning](#survey-data-cleaning)
- [Descriptive Statistics](#descriptive-statistics)
- [Likert Scale Analysis](#likert-scale-analysis)
- [Hypothesis Testing (T-tests/ANOVA)](#hypothesis-testing-t-testsanova)
- [Linear Regression Analysis](#linear-regression-analysis)
- [Logistic Regression](#logistic-regression)
- [Factor Analysis (PCA)](#factor-analysis-pca)
- [Cluster Analysis](#cluster-analysis)
- [Social Network Construction](#social-network-construction)
- [Network Centrality Measures](#network-centrality-measures)
- [Community Detection](#community-detection)
- [Time Series Analysis](#time-series-analysis)
- [Geospatial Mapping](#geospatial-mapping)
- [Automated Coding Assistant](#automated-coding-assistant)
- [Inter-rater Reliability](#inter-rater-reliability)
- [Basic Web Scraping](#basic-web-scraping)
- [API Data Collection](#api-data-collection)
- [Data Format Converter](#data-format-converter)
- [Missing Data Imputation](#missing-data-imputation)
- [Merge and Join Datasets](#merge-and-join-datasets)
- [Outlier Detection](#outlier-detection)
- [Causal Inference (Propensity Score)](#causal-inference-propensity-score)
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
- [Data Anonymization](#data-anonymization)
- [Multiple Imputation for Missing Data](#multiple-imputation-for-missing-data)
- [Exporting Results](#exporting-results)
- [Repeated Measures ANOVA](#repeated-measures-anova)
- [Mixed ANOVA (Split-Plot Design)](#mixed-anova-split-plot-design)
- [Signal Detection Theory (SDT)](#signal-detection-theory-sdt)
- [Reaction Time Analysis](#reaction-time-analysis)
- [Power Analysis](#power-analysis)
- [Reliability Analysis (Cronbach's Alpha)](#reliability-analysis-cronbach's-alpha)
- [Effect Size Calculation](#effect-size-calculation)
- [Chi-Square Test of Independence](#chi-square-test-of-independence)
- [Mediation Analysis with Bootstrapping](#mediation-analysis-with-bootstrapping)
- [Plotting Interaction Effects](#plotting-interaction-effects)

---

### WhatsApp Chat Analysis

**File:** `01_whatsapp_chat_analysis.ipynb`

**Description:** This notebook demonstrates how to parse, clean, and analyze exported WhatsApp chat data. We will use regular expressions to extract dates, times, senders, and messages, and then use pandas and matplotlib/seaborn for analysis and visualization.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/01_whatsapp_chat_analysis.ipynb)

---

### Sentiment Analysis

**File:** `02_sentiment_analysis.ipynb`

**Description:** This notebook performs sentiment analysis on text data using TextBlob and NLTK (VADER).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/02_sentiment_analysis.ipynb)

---

### Topic Modeling (LDA)

**File:** `03_topic_modeling.ipynb`

**Description:** This notebook demonstrates how to extract topics from a collection of documents using Latent Dirichlet Allocation (LDA) with Scikit-Learn.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/03_topic_modeling.ipynb)

---

### Word Cloud & N-Grams

**File:** `04_wordcloud_ngrams.ipynb`

**Description:** Visualizes frequent words and phrases using WordCloud and NLTK.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/04_wordcloud_ngrams.ipynb)

---

### Named Entity Recognition (NER)

**File:** `05_named_entity_recognition.ipynb`

**Description:** Identifies people, organizations, and locations in text using SpaCy or NLTK (using NLTK here for simplicity without heavy models).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/05_named_entity_recognition.ipynb)

---

### Hate Speech/Toxicity Detection

**File:** `06_hate_speech_detection.ipynb`

**Description:** Demonstrates a basic text classification pipeline for detecting toxic comments.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/06_hate_speech_detection.ipynb)

---

### Text Preprocessing Pipeline

**File:** `07_text_preprocessing.ipynb`

**Description:** Comprehensive guide to cleaning text (stopwords, lemmatization, etc.).

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/07_text_preprocessing.ipynb)

---

### Survey Data Cleaning

**File:** `08_survey_cleaning.ipynb`

**Description:** Techniques for handling missing values, recoding variables, and data type conversion.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/08_survey_cleaning.ipynb)

---

### Descriptive Statistics

**File:** `09_descriptive_stats.ipynb`

**Description:** Summary statistics and distribution plots.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/09_descriptive_stats.ipynb)

---

### Likert Scale Analysis

**File:** `10_likert_scale_analysis.ipynb`

**Description:** Visualizing ordinal data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/10_likert_scale_analysis.ipynb)

---

### Hypothesis Testing (T-tests/ANOVA)

**File:** `11_group_comparisons.ipynb`

**Description:** Comparing means between groups.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/11_group_comparisons.ipynb)

---

### Linear Regression Analysis

**File:** `12_linear_regression.ipynb`

**Description:** Predicting a continuous outcome variable based on one or more predictor variables.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/12_linear_regression.ipynb)

---

### Logistic Regression

**File:** `13_logistic_regression.ipynb`

**Description:** Predicting binary outcomes.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/13_logistic_regression.ipynb)

---

### Factor Analysis (PCA)

**File:** `14_factor_analysis.ipynb`

**Description:** Reducing dimensionality of data.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/14_factor_analysis.ipynb)

---

### Cluster Analysis

**File:** `15_cluster_analysis.ipynb`

**Description:** Segmenting data using K-Means.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/15_cluster_analysis.ipynb)

---

### Social Network Construction

**File:** `16_network_construction.ipynb`

**Description:** Creating nodes and edges from interaction data using NetworkX.

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

### Basic Web Scraping

**File:** `23_web_scraping.ipynb`

**Description:** Collecting data from websites using BeautifulSoup and Requests.

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

### Outlier Detection

**File:** `28_outlier_detection.ipynb`

**Description:** Identifying and handling outliers using Z-score.

**Status:** Verified ✅

[Open Notebook](social_science_research_notebooks/28_outlier_detection.ipynb)

---

### Causal Inference (Propensity Score)

**File:** `29_causal_inference.ipynb`

**Description:** Simple demonstration of logic for matching (requires causalinference lib usually, but showing logic here).

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

**Description:** Classifying text into predefined categories using TF-IDF and Linear SVM.

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

### Data Anonymization

**File:** `48_data_anonymization.ipynb`

**Description:** Basic techniques to protect privacy: Hashing identifiers and binning quasi-identifiers.

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

### Repeated Measures ANOVA

**File:** `51_repeated_measures_anova.ipynb`

**Description:** Analyzing within-subjects designs where the same subjects are measured under multiple conditions.

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
