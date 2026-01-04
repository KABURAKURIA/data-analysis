# Social Science Research Notebooks

This repository contains 60 Jupyter Notebooks designed to cover various aspects of social science research, including data cleaning, analysis, and visualization. These notebooks are designed to handle multiple forms of data.

## Notebooks List

### 1. WhatsApp Chat Analysis (Starter)
- **File:** `01_whatsapp_chat_analysis.ipynb`
- **Description:**  Parses exported WhatsApp text files, cleans the data, and provides visualizations of activity over time, user activity, and basic content analysis.

### Text Analysis & NLP
2. **Sentiment Analysis**: `02_sentiment_analysis.ipynb` - Analyzes sentiment (positive/negative/neutral) of text data.
3. **Topic Modeling (LDA)**: `03_topic_modeling.ipynb` - Extracts themes/topics from a collection of documents using Latent Dirichlet Allocation.
4. **Word Cloud & N-Grams**: `04_wordcloud_ngrams.ipynb` - Visualizes frequent words and phrases.
5. **Named Entity Recognition**: `05_named_entity_recognition.ipynb` - Identifies people, organizations, and locations in text.
6. **Hate Speech/Toxicity Detection**: `06_hate_speech_detection.ipynb` - Basic classification of toxic comments.
7. **Text Preprocessing Pipeline**: `07_text_preprocessing.ipynb` - comprehensive guide to cleaning text (stopwords, lemmatization, etc.).
39. **Document Similarity**: `39_document_similarity.ipynb` - Calculating Cosine Similarity between documents.
40. **Supervised Text Classification**: `40_text_classification.ipynb` - Classifying text into predefined categories using TF-IDF and Linear SVM.
44. **PDF Text Extraction**: `44_pdf_text_extraction.ipynb` - Extracting text from PDF files.

### Survey & Quantitative Analysis
8. **Survey Data Cleaning**: `08_survey_cleaning.ipynb` - Handling missing data, recoding variables, and data type conversion.
9. **Descriptive Statistics**: `09_descriptive_stats.ipynb` - Summary statistics and distribution plots for survey items.
10. **Likert Scale Analysis**: `10_likert_scale_analysis.ipynb` - Specific visualizations and correlations for ordinal data.
11. **Hypothesis Testing (T-tests/ANOVA)**: `11_group_comparisons.ipynb` - Comparing means between groups.
12. **Linear Regression**: `12_linear_regression.ipynb` - Predicting continuous outcomes.
13. **Logistic Regression**: `13_logistic_regression.ipynb` - Predicting binary outcomes.
14. **Factor Analysis (EFA/PCA)**: `14_factor_analysis.ipynb` - Reducing dimensionality of survey items.
15. **Cluster Analysis**: `15_cluster_analysis.ipynb` - Segmenting respondents into groups.
31. **Multilevel Modeling (HLM)**: `31_multilevel_modeling.ipynb` - Analyzing nested data (e.g., students in schools).
36. **Mediation Analysis**: `36_mediation_analysis.ipynb` - Testing if the effect of X on Y is mediated by M.
37. **Moderation Analysis**: `37_moderation_analysis.ipynb` - Testing if the relationship between X and Y depends on Z.
38. **Meta-Analysis Visualization**: `38_meta_analysis.ipynb` - Visualizing effect sizes using a Forest Plot.
58. **Chi-Square Test**: `58_chi_square_test.ipynb` - Tests of independence for categorical data.
59. **Mediation (Bootstrapped)**: `59_mediation_bootstrapped.ipynb` - Robust mediation analysis.
60. **Interaction Plots**: `60_interaction_plots.ipynb` - Visualizing 2-way interactions.

### Experimental Psychology
51. **Repeated Measures ANOVA**: `51_repeated_measures_anova.ipynb` - Within-subjects analysis.
52. **Mixed ANOVA**: `52_mixed_anova.ipynb` - Split-plot designs (Within x Between).
53. **Signal Detection Theory**: `53_signal_detection_theory.ipynb` - Calculating d-prime and criterion.
54. **Reaction Time Analysis**: `54_reaction_time_analysis.ipynb` - Outlier removal and transformations.
55. **Power Analysis**: `55_power_analysis.ipynb` - Sample size calculations.
56. **Reliability Analysis**: `56_reliability_analysis.ipynb` - Cronbach's Alpha.
57. **Effect Sizes**: `57_effect_sizes.ipynb` - Cohen's d, Eta-squared.

### Causal Inference & Advanced Stats
29. **Propensity Score Matching**: `29_causal_inference.ipynb` - Matching techniques for observational studies.
33. **Difference-in-Differences (DiD)**: `33_difference_in_differences.ipynb` - Policy evaluation technique.
34. **Regression Discontinuity (RDD)**: `34_regression_discontinuity.ipynb` - Causal inference near a cutoff.
35. **Fixed Effects (Panel Data)**: `35_fixed_effects_panel.ipynb` - Controlling for unobserved heterogeneity.
32. **Survival Analysis**: `32_survival_analysis.ipynb` - Analyzing time-to-event data.

### Machine Learning
41. **Decision Tree Classifier**: `41_decision_tree.ipynb` - Interpretable machine learning model.
42. **Random Forest Analysis**: `42_random_forest.ipynb` - Classification and feature importance.
43. **K-Nearest Neighbors (KNN)**: `43_knn_classification.ipynb` - Simple classification based on proximity.

### Network Analysis
16. **Social Network Construction**: `16_network_construction.ipynb` - Creating nodes and edges from interaction data.
17. **Network Centrality Measures**: `17_network_centrality.ipynb` - Calculating Degree, Betweenness, and Closeness centrality.
18. **Community Detection**: `18_community_detection.ipynb` - Identifying subgroups within a network.

### Time Series & Geospatial
19. **Time Series Analysis**: `19_time_series_analysis.ipynb` - Analyzing trends and seasonality.
20. **Geospatial Mapping**: `20_geospatial_mapping.ipynb` - visualizing data on maps using Folium/Geopandas.
46. **Advanced Date/Time Handling**: `46_advanced_datetime.ipynb` - Timezones and durations.

### Data Utilities & Collection
23. **Web Scraping Basic**: `23_web_scraping.ipynb` - Collecting data from websites.
24. **API Data Collection**: `24_api_data_collection.ipynb` - Example of fetching data from a public API.
25. **Data Format Converter**: `25_data_converter.ipynb` - Converting between JSON, Excel, CSV, and XML.
26. **Missing Data Imputation (Basic)**: `26_missing_data_imputation.ipynb` - Standard techniques.
49. **Multiple Imputation**: `49_multiple_imputation.ipynb` - Advanced imputation using MICE.
27. **Merge and Join Datasets**: `27_merge_datasets.ipynb` - Combining data from multiple sources.
28. **Outlier Detection**: `28_outlier_detection.ipynb` - Identifying and handling outliers.
45. **Nested JSON Flattening**: `45_json_flattening.ipynb` - Handling complex JSON structures.
48. **Data Anonymization**: `48_data_anonymization.ipynb` - Hashing and binning for privacy.
50. **Exporting Results**: `50_exporting_results.ipynb` - Saving to Excel with multiple sheets.

### Qualitative & Interactive
21. **Automated Coding Assistant**: `21_auto_coding.ipynb` - Keyword-based extraction.
22. **Inter-rater Reliability**: `22_inter_rater_reliability.ipynb` - Calculating Cohen's Kappa.
47. **Interactive Widgets**: `47_interactive_widgets.ipynb` - Interactive data exploration.

### Evaluation
30. **Model Evaluation Metrics**: `30_model_evaluation.ipynb` - Confusion matrices, ROC curves.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter Lab/Notebook: `jupyter lab`
3. Navigate to the desired notebook.
