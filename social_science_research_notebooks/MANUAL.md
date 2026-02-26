# Social Science Research Notebooks Manual

This manual is automatically generated from the notebooks in this repository.

---

## Notebook: `01_whatsapp_chat_analysis.ipynb`

# WhatsApp Chat Analysis In-Depth

This notebook provides a comprehensive toolkit for analyzing exported WhatsApp chat data. Unlike basic tutorials, this notebook covers:

1.  **Robust Data Generation**: Creating a realistic synthetic dataset for testing.
2.  **Advanced Parsing**: Handling multiple device formats (iOS/Android) and multi-line messages.
3.  **Temporal Analysis**: Activity heatmaps (Day x Hour) and response time distributions.
4.  **Content Analysis**: Sentiment tracking, Emoji frequency, and Word Clouds.
5.  **Interaction Network**: visualizing who replies to whom.

## Libraries Used
- `pandas`: Data manipulation.
- `seaborn` / `matplotlib`: Visualization.
- `emoji`: Extracting and counting emojis.
- `textblob`: Sentiment analysis.
- `wordcloud`: Visualizing frequent terms.

---

## Notebook: `02_sentiment_analysis.ipynb`

# Sentiment Analysis In-Depth
This notebook provides a deep dive into sentiment analysis using two popular libraries: TextBlob and NLTK (VADER). We will compare their performance on a comprehensive synthetic dataset of movie reviews, clean the text, and visualize the results.

---

## Notebook: `03_topic_modeling.ipynb`

# Topic Modeling (LDA) In-Depth

Topic modeling is an unsupervised machine learning technique used to discover abstract "topics" that occur in a collection of documents. This notebook demonstrates how to use **Latent Dirichlet Allocation (LDA)** with Scikit-Learn to extract themes from a synthetic dataset of research abstracts.

## Objectives
1.  Generate a synthetic dataset with distinct underlying themes.
2.  Preprocess text data (Tokenization, Stopword Removal).
3.  Vectorize text using CountVectorizer (Bag of Words).
4.  Fit an LDA model.
5.  Visualize and interpret the topics.

---

## Notebook: `04_wordcloud_ngrams.ipynb`

# Word Cloud & N-Grams In-Depth

This notebook explores text visualization techniques beyond simple frequency counts. We will generate a larger synthetic dataset, preprocess it, and visualize not just single words (unigrams) but also pairs (bigrams) and triplets (trigrams) to understand context.

---

## Notebook: `05_named_entity_recognition.ipynb`

# Named Entity Recognition (NER) In-Depth

NER involves identifying and classifying named entities (people, organizations, locations, dates) in text. This notebook compares NLTK (rule-based/statistical) and SpaCy (model-based) approaches, and visualizes the results on a synthetic news article.

---

## Notebook: `06_hate_speech_detection.ipynb`

# Hate Speech / Toxicity Detection In-Depth

This notebook demonstrates a supervised learning pipeline for detecting toxic comments. We will generate a synthetic dataset, extract features using TF-IDF, train a Logistic Regression model, and evaluating its performance using industry-standard metrics.

---

## Notebook: `07_text_preprocessing.ipynb`

# Text Preprocessing Pipeline In-Depth

Preprocessing is the foundation of NLP. A bad cleaning pipeline creates "garbage in, garbage out". This notebook demonstrates a robust pipeline handling URLs, HTML tags, emojis, and compares Stemming vs. Lemmatization.

---

## Notebook: `08_survey_cleaning.ipynb`

# Survey Data Cleaning: An In-Depth Guide

Real-world survey data is rarely clean. It contains duplicates, inconsistent spelling, outliers, and missing values. This notebook demonstrates a professional workflow for cleaning a messy dataset using `pandas`.

## Learning Objectives
1.  **Detect and remove duplicates.**
2.  **Standardize categorical variables** (e.g., mapping "m", "Male", "M" to "Male").
3.  **Handle missing data** effectively (Drop vs Impute).
4.  **Detect outliers** in numeric responses.
5.  **Recode reverse-coded items** (common in psychological surveys).

---

## Notebook: `09_descriptive_stats.ipynb`

# Descriptive Statistics In-Depth

This notebook goes beyond basic summaries to explore robust statistics, distribution shapes, and grouped analyses using a realistic synthetic dataset.

---

## Notebook: `10_likert_scale_analysis.ipynb`

# Likert Scale Analysis In-Depth

This notebook focuses on analyzing and visualizing survey data collected on a Likert scale (e.g., Strongly Disagree to Strongly Agree). We will cover reliability analysis and specialized visualizations.

---

## Notebook: `11_group_comparisons.ipynb`

# Group Comparisons: T-tests & ANOVA In-Depth

A fundamental task in social science is comparing means between groups (e.g., Treatment vs Control). This notebook goes beyond simple p-values to include:

1.  **Assumption Checking**: Normality and Homogeneity of Variance.
2.  **Effect Size**: Cohen's d (t-test) and Eta-squared (ANOVA).
3.  **Visualization**: Boxplots with jitter points to show raw data.
4.  **Post-hoc Testing**: Determining exactly *which* groups differ.

---

## Notebook: `12_linear_regression.ipynb`

# Linear Regression Analysis In-Depth

This notebook covers linear regression with a focus on statistical inference and assumption checking, which is crucial for social science research. We will use `statsmodels` for detailed output.

---

## Notebook: `13_logistic_regression.ipynb`

# Logistic Regression In-Depth

Predicting binary outcomes (Voting Behavior) using Logistic Regression. We will use `statsmodels` for statistical inference (Odds Ratios) and `scikit-learn` for predictive evaluation.

---

## Notebook: `14_factor_analysis.ipynb`

# Exploratory Factor Analysis (EFA) In-Depth
This notebook demonstrates how to perform Exploratory Factor Analysis (EFA) to uncover latent structures in survey data. We use the `factor_analyzer` library to identify underlying factors (e.g., personality traits) from a set of observed variables (Likert items).

---

## Notebook: `15_cluster_analysis.ipynb`

# Cluster Analysis In-Depth

Cluster analysis is an unsupervised machine learning technique used to group similar data points together. In social science, this is often used for **segmentation**—identifying distinct subgroups within a population (e.g., voters, consumers, survey respondents) without predefined labels.

In this notebook, we will:
1.  Generate a realistic synthetic dataset of **Online Shoppers** (Age, Income, Spending Score).
2.  Preprocess the data (Standardization).
3.  Determine the optimal number of clusters using the **Elbow Method** and **Silhouette Analysis**.
4.  Perform **K-Means Clustering**.
5.  Visualize and interpret the resulting segments.

---

## Notebook: `16_network_construction.ipynb`

# Social Network Construction & Analysis In-Depth

This notebook moves beyond basic graph creation to explore complex network analysis techniques using real-world-like data. We will analyze the famous Zachary's Karate Club graph and simulate a larger scale-free network.

---

## Notebook: `17_network_centrality.ipynb`

# Network Centrality Measures In-Depth

Centrality measures allow us to identify the most important nodes in a network. In social networks, these might be influencers, bridges between communities, or efficient communicators.

In this notebook, we will:
1.  Generate a **Synthetic Social Network** (Scale-Free) using the Barabási-Albert model.
2.  Assign fictitious identities to nodes using `Faker`.
3.  Calculate key centrality measures:
    *   **Degree Centrality**: Popularity (number of connections).
    *   **Betweenness Centrality**: Control of information flow (bridges).
    *   **Closeness Centrality**: Efficiency (average distance to others).
    *   **Eigenvector Centrality**: Influence (connected to other well-connected nodes).
4.  Visualize the network highlighting influential nodes.

---

## Notebook: `18_community_detection.ipynb`

# Community Detection In-Depth

Identifying subgroups within a network using Louvain or Greedy Modularity.

---

## Notebook: `19_time_series_analysis.ipynb`

# Time Series Analysis In-Depth

Analyzing trends and seasonality in time-series data using classical statistical methods.

---

## Notebook: `20_geospatial_mapping.ipynb`

# Geospatial Mapping In-Depth

Visualizing data on maps.

---

## Notebook: `21_auto_coding.ipynb`

# Automated Coding Assistant In-Depth

Keyword-based extraction to assist qualitative coding.

---

## Notebook: `22_inter_rater_reliability.ipynb`

# Inter-rater Reliability In-Depth

Calculating Cohen's Kappa.

---

## Notebook: `23_web_scraping.ipynb`

# Web Scraping for Social Science: An In-Depth Guide

Web scraping is a powerful tool for collecting digital trace data. This notebook moves beyond simple tutorials to cover robust, ethical, and structured data collection.

## Key Concepts Covered
1.  **Ethical Scraping**: `robots.txt` and rate limiting.
2.  **HTML Parsing**: Navigating the DOM with `BeautifulSoup`.
3.  **Pagination**: looping through multiple pages.
4.  **Error Handling**: Managing timeouts and missing elements.
5.  **Data Structuring**: Saving results to a clean CSV.

---

## Notebook: `24_api_data_collection.ipynb`

# API Data Collection In-Depth

Fetching data from a public API.

---

## Notebook: `25_data_converter.ipynb`

# Data Format Converter In-Depth

Converting between JSON, CSV, and Excel formats.

---

## Notebook: `26_missing_data_imputation.ipynb`

# Missing Data Imputation In-Depth

Advanced techniques for handling missing values using KNN.

---

## Notebook: `27_merge_datasets.ipynb`

# Merge and Join Datasets In-Depth

Combining data from multiple sources.

---

## Notebook: `28_outlier_detection.ipynb`

# Outlier Detection In-Depth

Identifying and handling outliers is crucial in social science research. We explore three methods:
1.  **Z-Score**: Parametric method (assumes normality).
2.  **IQR Method**: Non-parametric method (robust to extreme values).
3.  **Isolation Forest**: Machine Learning method for high-dimensional or complex data.

---

## Notebook: `29_causal_inference.ipynb`

# Causal Inference: Propensity Score Matching (PSM) In-Depth

In observational studies, we cannot randomly assign treatment. This leads to **selection bias**: the treated group might be fundamentally different from the control group.

**Propensity Score Matching (PSM)** attempts to mimic a randomized control trial (RCT) by matching treated units with control units that have similar probabilities of receiving the treatment (Propensity Scores).

---

## Notebook: `30_model_evaluation.ipynb`

# Model Evaluation Metrics In-Depth

Confusion matrices, ROC curves.

---

## Notebook: `31_multilevel_modeling.ipynb`

# Multilevel Modeling (Hierarchical Linear Models) In-Depth

Analyzing data where observations are nested within groups (e.g., students within schools).

---

## Notebook: `32_survival_analysis.ipynb`

# Survival Analysis In-Depth

Analyzing time-to-event data (e.g., time until employment, customer churn) using Lifelines.

---

## Notebook: `33_difference_in_differences.ipynb`

# Difference-in-Differences (DiD) In-Depth

Policy evaluation technique comparing treatment and control groups over time.

---

## Notebook: `34_regression_discontinuity.ipynb`

# Regression Discontinuity Design (RDD) In-Depth

Causal inference near a cutoff point.

---

## Notebook: `35_fixed_effects_panel.ipynb`

# Fixed Effects Model (Panel Data) In-Depth

Controlling for time-invariant unobserved heterogeneity.

---

## Notebook: `36_mediation_analysis.ipynb`

# Mediation Analysis In-Depth

Testing if the effect of X on Y is mediated by M (X -> M -> Y).

---

## Notebook: `37_moderation_analysis.ipynb`

# Moderation Analysis In-Depth

Testing if the relationship between X and Y depends on Z (Interaction).

---

## Notebook: `38_meta_analysis.ipynb`

# Meta-Analysis Visualization In-Depth

Visualizing effect sizes from multiple studies using a Forest Plot.

---

## Notebook: `39_document_similarity.ipynb`

# Document Similarity In-Depth

Calculating Cosine Similarity between documents.

---

## Notebook: `40_text_classification.ipynb`

# Supervised Text Classification In-Depth

Classifying text into predefined categories using TF-IDF with Linear SVM and Naive Bayes.

---

## Notebook: `41_decision_tree.ipynb`

# Decision Tree Classifier In-Depth

Interpretable machine learning model.

---

## Notebook: `42_random_forest.ipynb`

# Random Forest Analysis In-Depth

Using Random Forest for classification and feature importance.

---

## Notebook: `43_knn_classification.ipynb`

# K-Nearest Neighbors (KNN) In-Depth

Simple classification based on proximity.

---

## Notebook: `44_pdf_text_extraction.ipynb`

# PDF Text Extraction In-Depth

Extracting text from PDF files using pypdf. We first create a dummy PDF.

---

## Notebook: `45_json_flattening.ipynb`

# Nested JSON Flattening In-Depth

Handling complex JSON structures often found in social media API data.

---

## Notebook: `46_advanced_datetime.ipynb`

# Advanced Date and Time Handling In-Depth

Handling timezones and calculating durations.

---

## Notebook: `47_interactive_widgets.ipynb`

# Interactive Widgets In-Depth

Using ipywidgets for interactive data exploration.

---

## Notebook: `48_data_anonymization.ipynb`

# Data Anonymization & Privacy In-Depth

Protecting participant privacy is a core ethical requirement in social science. This notebook demonstrates techniques for de-identifying data, including hashing, k-anonymity, and differential privacy concepts.

## Key Concepts
1.  **Direct Identifiers**: Names, SSNs, Emails (Must be removed/hashed).
2.  **Quasi-Identifiers (QI)**: Age, Zip Code, Gender (Can be combined to re-identify).
3.  **K-Anonymity**: Ensuring any record is indistinguishable from at least *k-1* other records.

---

## Notebook: `49_multiple_imputation.ipynb`

# Multiple Imputation for Missing Data In-Depth

Using IterativeImputer (MICE) to handle missing values.

---

## Notebook: `50_exporting_results.ipynb`

# Exporting Results In-Depth

Saving DataFrames to nicely formatted Excel files.

---

## Notebook: `51_repeated_measures_anova.ipynb`

# Repeated Measures ANOVA In-Depth

This notebook performs a rigorous Repeated Measures ANOVA using `pingouin`. It includes data generation, assumption checking (Sphericity, Normality), and effect size calculation.

---

## Notebook: `52_mixed_anova.ipynb`

# Mixed ANOVA (Split-Plot Design) In-Depth

Analyzing designs with both within-subjects and between-subjects factors.

---

## Notebook: `53_signal_detection_theory.ipynb`

# Signal Detection Theory (SDT) In-Depth

Calculating sensitivity ($d'$) and response bias ($c$ or $\beta$).

---

## Notebook: `54_reaction_time_analysis.ipynb`

# Reaction Time Analysis In-Depth

Preprocessing RT data: Outlier removal, transformations, and visualization.

---

## Notebook: `55_power_analysis.ipynb`

# Power Analysis In-Depth

Calculating required sample size or achieved power using statsmodels and Pingouin.

---

## Notebook: `56_reliability_analysis.ipynb`

# Reliability Analysis (Cronbach's Alpha) In-Depth

Assessing the internal consistency of a scale/survey.

---

## Notebook: `57_effect_sizes.ipynb`

# Effect Size Calculation In-Depth

Calculating Cohen's d, Hedges' g, and Eta-squared.

---

## Notebook: `58_chi_square_test.ipynb`

# Chi-Square Test of Independence In-Depth

Analyzing the relationship between two categorical variables.

---

## Notebook: `59_mediation_bootstrapped.ipynb`

# Mediation Analysis with Bootstrapping In-Depth

A more robust method for testing indirect effects using Pingouin.

---

## Notebook: `60_interaction_plots.ipynb`

# Plotting Interaction Effects In-Depth

Visualizing 2-way and 3-way interactions using Seaborn.

---

## Notebook: `61_data_analyst_roadmap.ipynb`

# Data Analyst Roadmap & Guide In-Depth

This notebook serves as a comprehensive guide through the Data Analyst Roadmap, mapping key concepts to practical Python implementations found within this repository. It is aligned with industry standard roadmaps (e.g., [roadmap.sh/data-analyst](https://roadmap.sh/data-analyst)) and covers everything from Excel foundations to Advanced Machine Learning.

**Roadmap Sections:**
1.  Introduction & Types of Analytics
2.  Building a Strong Foundation (Excel Equivalents)
3.  Programming Languages (Python vs R)
4.  Data Collection (Databases, APIs, Scraping)
5.  Data Cleanup (Handling Missing Data, Outliers)
6.  Exploratory Data Analysis (EDA) & Visualization
7.  Statistical Analysis
8.  Advanced Topics (Machine Learning, Big Data, Deep Learning)
9.  Projects & Portfolio
10. Online Courses & Certifications
11. Soft Skills & Networking


---

## Notebook: `62_sql_for_analysts.ipynb`

# SQL for Data Analysts In-Depth

Structured Query Language (SQL) is the standard language for managing and manipulating relational databases. For a Data Analyst, it is often considered the most critical technical skill alongside Python/R.

In this notebook, we will:
1.  Create a temporary SQLite database in memory.
2.  Populate it with synthetic social science data (Participants and Survey Responses).
3.  Perform essential SQL operations: `SELECT`, `WHERE`, `JOIN`, `GROUP BY`.
4.  Explore advanced concepts: **Window Functions** and **CTEs**.
5.  Integrate SQL with Python using `pandas`.


---

## Notebook: `63_advanced_visualization_plotly.ipynb`

# Advanced Visualization with Plotly In-Depth

While `matplotlib` and `seaborn` are excellent for static publications, `plotly` allows for interactive data exploration (zooming, hovering, filtering).

In this notebook, we will create:
1.  Interactive Scatter Plots (with hover data)
2.  Interactive Line Charts
3.  Box Plots with points
4.  3D Scatter Plots
5.  Funnel Charts


---

## Notebook: `64_intro_to_transformers.ipynb`

# Introduction to Transformers for NLP In-Depth

The Transformer architecture (e.g., BERT, GPT) has revolutionized Natural Language Processing (NLP). Unlike older methods (TF-IDF, simple RNNs), Transformers capture deep contextual relationships in text.

In this notebook, we will use the **Hugging Face `transformers` library** to perform state-of-the-art NLP tasks with just a few lines of code.

**Tasks:**
1.  **Sentiment Analysis**: Determining if a text is positive or negative.
2.  **Zero-Shot Classification**: Classifying text into categories the model hasn't explicitly seen during training.


---

## Notebook: `65_bayesian_analysis_pymc.ipynb`

# Bayesian Analysis with PyMC In-Depth

Bayesian statistics offers a powerful alternative to frequentist (classic) statistics. Instead of calculating a single 'p-value', Bayesian analysis allows us to estimate the **probability distribution** of our parameters given the data.

In this notebook, we will use **PyMC** (formerly PyMC3), the standard library for Probabilistic Programming in Python, to perform a simple Bayesian Linear Regression.

**Key Concepts:**
1.  **Prior**: What we believe about the parameters *before* seeing data.
2.  **Likelihood**: How likely the data is given the parameters.
3.  **Posterior**: What we believe about the parameters *after* seeing data (Prior + Likelihood).


---

## Notebook: `66_rag_for_research.ipynb`

# Retrieval-Augmented Generation (RAG) for Research In-Depth

**RAG** is a technique to improve the accuracy of Large Language Models (LLMs) by giving them access to private or specific data. Instead of relying solely on what the model 'knows' from its training, we **retrieve** relevant documents and feed them to the model as context.

This is incredibly useful for qualitative researchers who want to "chat" with their interview transcripts or literature review.

**Steps:**
1.  **Embed**: Convert text documents into numbers (vectors).
2.  **Index**: Store vectors in a searchable database (FAISS).
3.  **Retrieve**: Find the most relevant documents for a user question.
4.  **Generate**: Ask an LLM to answer the question using the retrieved text.


---

## Notebook: `67_dashboard_with_panel.ipynb`

# Building Interactive Dashboards with Panel In-Depth

**Panel** is a powerful library that lets you create custom interactive web apps and dashboards by connecting user-defined widgets to plots, images, tables, or text.

Unlike Streamlit (which reruns the whole script), Panel is designed to work well within Jupyter Notebooks and can also be deployed as a standalone app.

In this notebook, we will build a dashboard to explore a synthetic dataset of **Public Opinion Polling**.

---

## Notebook: `68_instrumental_variables.ipynb`

# Causal Inference: Instrumental Variables (IV) In-Depth

One of the biggest challenges in social science is **Endogeneity** (Omitted Variable Bias). This happens when there is a hidden variable that affects both your treatment (cause) and your outcome (effect).

For example, let's say we want to study the effect of **Education** on **Income**.
- Problem: **Ability** (unobserved) affects both Education and Income.
- Result: Naive regression overestimates the return to education.

**Instrumental Variables (IV)** allow us to isolate the causal effect by using a third variable (Z) that affects Education (X) but has *no direct effect* on Income (Y) except through Education.


---

## Notebook: `69_python_basics_for_social_scientists.ipynb`

# Python In-Depth for Social Scientists

This notebook covers the fundamental building blocks of Python programming: Variables, Data Structures (Lists, Dictionaries), Loops, and Functions. We will use social science examples (e.g., ages of participants) to make it practical.

---

## Notebook: `70_pandas_manipulation_basics.ipynb`

# Pandas Manipulation In-Depth

Pandas is the most powerful tool for tabular data manipulation in Python. Think of it as 'Excel on steroids'. This notebook covers DataFrames, Filtering, and Aggregation.

---

## Notebook: `71_data_visualization_101.ipynb`

# Data Visualization In-Depth Guide

This notebook introduces `matplotlib` and `seaborn` for creating publication-quality plots.

---

## Notebook: `72_advanced_ml_xgboost.ipynb`

# Advanced ML: Gradient Boosting with XGBoost In-Depth

XGBoost (eXtreme Gradient Boosting) is one of the most popular and efficient algorithms for structured (tabular) data. It often outperforms other algorithms like Random Forests.

---

## Notebook: `73_neural_networks_tabular.ipynb`

# Neural Networks for Tabular Data (PyTorch) In-Depth

While Deep Learning is famous for Images and NLP, it can also be used for tabular data. This notebook demonstrates how to build a simple Multi-Layer Perceptron (MLP) using PyTorch.

---

## Notebook: `74_exploratory_data_analysis.ipynb`

# Exploratory Data Analysis (EDA) In-Depth

A comprehensive workflow for inspecting, cleaning, and understanding your data before modeling.

---

## Notebook: `77_audio_transcription.ipynb`

# Audio Transcription with Whisper In-Depth
This notebook demonstrates how to transcribe audio using OpenAI's Whisper model and generate speech using gTTS.

---

## Notebook: `78_image_analysis_basics.ipynb`

# Image Analysis In-Depth
This notebook demonstrates how to load, process, and classify images using PyTorch and Torchvision.

---

## Notebook: `79_sem_structural_equation_modeling.ipynb`

# Structural Equation Modeling (SEM) In-Depth
This notebook demonstrates how to perform Structural Equation Modeling using `semopy`.

---

## Notebook: `80_ab_testing_basics.ipynb`

# A/B Testing and Power Analysis In-Depth
This notebook demonstrates how to design and analyze A/B tests using Python.

---

## Notebook: `81_conjoint_analysis.ipynb`

# Conjoint Analysis (Rating-Based) In-Depth
This notebook demonstrates how to perform a Rating-Based Conjoint Analysis. We will generate synthetic data representing consumer preferences for a product (e.g., a Smartphone) with varying attributes (Price, Brand, Storage) and estimate the utility of each attribute level using OLS regression.

---

## Notebook: `82_fairness_audit_in_ai.ipynb`

# Algorithmic Fairness Audit In-Depth

As social scientists, we are often concerned with inequality and bias. When Machine Learning models are used to make decisions (e.g., hiring, loans, policing), they can reproduce or even amplify existing social biases.

In this notebook, we will:
1.  Generate a synthetic "Hiring" dataset with intentional gender bias.
2.  Train a standard Logistic Regression model.
3.  Audit the model using `fairlearn` and manual metrics.
4.  Explore mitigation strategies.

---

## Notebook: `83_advanced_topic_modeling_bertopic.ipynb`

# Advanced Topic Modeling with BERTopic In-Depth

Latent Dirichlet Allocation (LDA) (covered in notebook 03) is the classic method for topic modeling. However, it relies on a "Bag-of-Words" approach, ignoring the semantic meaning of words.

**BERTopic** is a modern topic modeling technique that leverages **Transformers** (BERT embeddings) to understand the context of words. It works by:
1.  Converting documents to vectors (Embeddings).
2.  Reducing dimensionality (UMAP).
3.  Clustering vectors (HDBSCAN).
4.  Extracting keywords (c-TF-IDF).

---

## Notebook: `84_handling_large_datasets.ipynb`

# Handling Large Datasets In-Depth

A common hurdle in social science research is the "Out of Memory" error. If you have a 10GB dataset and only 8GB of RAM, you cannot open it with standard `pd.read_csv()`.

In this notebook, we explore strategies to handle large data:
1.  **Chunking**: Processing data in small pieces.
2.  **Dask**: Parallel computing with lazy evaluation.
3.  **Efficient Formats**: Using Parquet instead of CSV.

---

## Notebook: `85_multimodal_analysis.ipynb`

# Multimodal Analysis: Text and Image In-Depth

**Description:** This notebook demonstrates **Multimodal Analysis** using the **CLIP (Contrastive Language-Image Pre-Training)** model via the Hugging Face `transformers` library. Multimodal analysis involves processing and correlating data from different modalities, such as text and images. This is increasingly relevant in social science for analyzing social media content (memes, posts with images).

We will cover:
1.  **Zero-Shot Image Classification**: Classifying an image into categories without training a specific model for those categories.
2.  **Image-Text Matching**: Finding the best caption for an image from a list of options.
3.  **Synthetic Image Generation**: Creating simple images with text to test the model's capabilities.

---

## Notebook: `86_recommender_systems_basics.ipynb`

# Recommender Systems In-Depth

Recommender systems are algorithms designed to suggest relevant items to users (e.g., movies to watch, books to read). They are fundamental to modern web applications.

In this notebook, we will explore two primary techniques using a synthetic dataset of **Books** and **User Ratings**:

1.  **Content-Based Filtering**: Recommends items similar to those a user liked, based on item attributes (e.g., Book Descriptions).
2.  **Collaborative Filtering**: Recommends items based on the past interactions of similar users (e.g., Matrix Factorization).

---

## Notebook: `87_explainable_ai_shap.ipynb`

# Explainable AI (XAI) with SHAP In-Depth

**Description:** This notebook explores **Explainable AI (XAI)** using the **SHAP (SHapley Additive exPlanations)** library. Machine learning models, especially complex ones like Random Forests or Neural Networks, are often "black boxes." SHAP values help us understand *why* a model made a specific prediction by quantifying the contribution of each feature.

We will cover:
1.  **Synthetic Data Generation**: Creating a realistic "Credit Risk" dataset.
2.  **Model Training**: Training a Random Forest Classifier to predict default risk.
3.  **SHAP Explainer**: calculating SHAP values.
4.  **Global Interpretability**: Summary plots to see overall feature importance.
5.  **Local Interpretability**: Force plots to explain individual predictions.
6.  **Dependence Plots**: Understanding interaction effects between features.

---

## Notebook: `88_time_series_forecasting_prophet.ipynb`

# Time Series Forecasting with Prophet In-Depth

**Description:** This notebook demonstrates time series forecasting using **Facebook Prophet**. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data.

We will cover:
1.  **Synthetic Data Generation**: Creating a time series with trend and seasonality (e.g., website traffic).
2.  **Model Fitting**: Training the Prophet model.
3.  **Forecasting**: Predicting future values.
4.  **Component Analysis**: Decomposing the forecast into trend and seasonal components.

---

## Notebook: `89_qualitative_analysis_with_llms.ipynb`

# Qualitative Analysis with LLMs (Zero-Shot Classification) In-Depth

Traditional qualitative coding is time-consuming. Large Language Models (LLMs) can automate the initial coding pass using Zero-Shot Classification. This notebook demonstrates how to categorize synthetic interview transcripts into themes without training a model.

**Techniques:**
1.  **Synthetic Interview Generation**: Using `Faker` to simulate qualitative data.
2.  **Zero-Shot Classification**: Using Hugging Face `transformers` to assign themes.
3.  **Visualization**: Comparing predicted themes.

---

## Notebook: `90_agentic_ai_concepts.ipynb`

# Agentic AI Concepts: The ReAct Pattern In-Depth

Agentic AI goes beyond simple classification or prediction. Agents can *reason* about a problem, *act* by using tools, and *observe* the results to adjust their plan. This pattern is known as **ReAct (Reasoning + Acting)**.

In this notebook, we build a **conceptual Agent** from scratch. Instead of using a paid API (like OpenAI), we use a "Mock LLM"—a simple function that simulates how a language model would respond to prompts—to demonstrate the architecture.

---

## Notebook: `91_imbalanced_learning.ipynb`

# Handling Imbalanced Datasets (SMOTE & ADASYN) In-Depth

In social science, interesting events are often rare: protests, fraud, hate speech, or disease outbreaks. Standard machine learning models bias towards the majority class, often ignoring the minority class entirely.

This notebook demonstrates how to solve this using **Resampling Techniques** from the `imbalanced-learn` library.

---

## Notebook: `92_bipartite_network_analysis.ipynb`

# Bipartite Network Analysis (Affiliation Networks) In-Depth

Most social networks we study are "One-Mode" (Person-Person). However, data often comes in "Two-Mode" or **Bipartite** forms: People attending Events, Directors on Boards, or Authors on Papers.

This notebook demonstrates how to handle these networks using `networkx`, specifically projecting them into one-mode networks to study the relationships implied by shared affiliations.

---

## Notebook: `93_synthetic_control_methods.ipynb`

# Synthetic Control Methods (SCM) In-Depth

The **Synthetic Control Method (SCM)** is a statistical method used to evaluate the effect of an intervention in comparative case studies. It involves constructing a weighted combination of control units (the "synthetic control") that best resembles the characteristics of the treated unit prior to the intervention.

## Scenario
Imagine a city ("City A") implements a new tax policy in year 2010. We want to know the effect of this tax on **GDP per capita**. We have data for City A and 10 other cities that did *not* implement the tax.

Instead of just comparing City A to the average of the other cities, we create a "Synthetic City A" which is a weighted average of the other cities, chosen to match City A's GDP trajectory *before* 2010.


---

## Notebook: `94_detecting_ai_generated_text.ipynb`

# Detecting AI-Generated Text In-Depth

As Large Language Models (LLMs) become ubiquitous, distinguishing between human and machine-generated text is a critical challenge in social science research.

Common hypotheses:
1.  **Perplexity (PPL)**: AI models aim to maximize probability, so they often choose "safe", predictable words. This leads to *lower* perplexity than human text.
2.  **Burstiness**: Humans write with varying sentence structures (short, then long and complex). AI tends to be more monotonic. Burstiness measures the variation in perplexity.

In this notebook, we will use a small model (**GPT-2**) to calculate these metrics on sample texts.


---

## Notebook: `95_graph_neural_networks.ipynb`

# Graph Neural Networks (GNN) In-Depth

This notebook demonstrates how to build a Graph Convolutional Network (GCN) from scratch using PyTorch to perform node classification on a synthetic Academic Citation Network.

---

## Notebook: `96_polars_data_analysis.ipynb`

# High-Performance Data Analysis with Polars

Polars is a lightning-fast DataFrame library implemented in Rust and memory-efficient. This notebook demonstrates its syntax and performance advantages over Pandas for large datasets.

---

## Notebook: `97_data_quality_validation.ipynb`

# Data Quality Validation with Pandera

In social science research, data quality is paramount. Manual checks are prone to error. `pandera` allows you to define a **schema** for your data and automatically validate it, catching errors early.

This notebook covers:
1.  **Generating 'Dirty' Data**: Simulating real-world errors (typos, out-of-range values).
2.  **Defining a Schema**: specifying expected data types and constraints.
3.  **Validation**: Running checks and handling failures.
4.  **Cleaning & Re-validating**: Fixing the data to pass the schema.

---

## Notebook: `98_local_llms_for_research.ipynb`

# Local LLMs for Research (No API Needed)

Using proprietary APIs (like OpenAI) can be costly and raise privacy concerns. This notebook demonstrates how to run **local** Large Language Models (LLMs) on your own machine (or Colab) using the Hugging Face `transformers` library.

We use `google/flan-t5-small`, a lightweight model capable of summarization, translation, and Q&A, which runs efficiently even on a CPU.

---

## Notebook: `99_spatial_autocorrelation.ipynb`

# Spatial Autocorrelation (Moran's I)

Spatial autocorrelation measures how correlated a variable is with itself in space.
-   **Positive**: High values cluster near high values (Hotspots).
-   **Negative**: High values are near low values (Checkerboard).
-   **Zero**: Random distribution.

We will use **Moran's I** to quantify this on a synthetic grid.

---

## Notebook: `100_capstone_project.ipynb`

# Capstone Project: End-to-End Social Media Analysis

This notebook represents a comprehensive workflow, taking you from raw data generation to advanced causal inference. It integrates skills learned throughout the previous notebooks.

**Scenario:**
You are a social scientist analyzing the impact of a new digital well-being policy introduced on **July 1st, 2023**.
*   **Treatment Group**: Users in the "South" region (where the policy was enforced).
*   **Control Group**: Users in the "North" region (no policy).
*   **Outcome**: Sentiment of social media posts.

**Steps:**
1.  **Data Simulation**: Generating realistic text and metadata.
2.  **Preprocessing**: Cleaning text and extracting sentiment.
3.  **EDA**: Visualizing trends over time.
4.  **Causal Inference**: Using Difference-in-Differences (DiD) to estimate the policy's effect.

---
