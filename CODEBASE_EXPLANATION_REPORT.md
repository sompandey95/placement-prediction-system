# Placement Prediction System Codebase Explanation Report

## 1. Executive Summary

This project is a Streamlit-based machine learning application for predicting whether an engineering student is likely to be placed. It has two user-facing experiences:

- A `Student Prediction` page for single-student prediction, explanation, roadmap generation, and PDF exports.
- A `T&P Dashboard` page for Training and Placement teams to analyze uploaded datasets and run batch predictions.

At a high level, the system follows this lifecycle:

1. A training pipeline reads a cleaned CSV dataset, preprocesses it, balances the training set with SMOTE, evaluates several classifiers, chooses the best one by weighted F1 score, optionally tunes it, and saves artifacts.
2. The Streamlit app loads those saved artifacts.
3. The student page converts live form input into the trained feature format, predicts placement probability, explains the prediction with SHAP, and optionally generates an AI roadmap and PDFs.
4. The T&P page either visualizes uploaded placement data or performs batch inference on many students at once.

This codebase is small, direct, and modular. The separation is mainly by responsibility:

- `train.py`: offline model training and artifact generation
- `src/`: reusable business logic and utilities
- `views/`: Streamlit UI pages
- `app.py`: app entry point and page router
- `data/` and `model/`: static input data and generated artifacts

## 2. Tech Stack

### Core language and framework

- Python
- Streamlit for the web UI

### Data and ML

- `pandas` for tabular data loading and transformation
- `scikit-learn` for preprocessing utilities, model training, evaluation, and cross-validation
- `xgboost` for gradient-boosted tree classification
- `imbalanced-learn` for SMOTE oversampling
- `joblib` for model serialization
- `shap` for local model explainability

### Visualization

- `plotly` for gauges, radar charts, pie charts, histograms, box plots, and heatmaps
- `matplotlib` is imported in `views/student.py`, but not meaningfully used in the current code

### Document generation

- `fpdf2` for PDF report generation

### AI integration

- `openai` Azure OpenAI client, wrapped in `src/advisor.py`
- `python-dotenv` for reading `.env` variables

### Configuration and metadata

- `PyYAML` for `config.yaml`
- JSON artifacts in `model/` for saved feature columns and model comparison results

### Notable dependencies currently present but not actively used by the shown code paths

- `SQLAlchemy`
- `psycopg2-binary`
- `openpyxl`
- `seaborn`
- Some Streamlit ecosystem packages installed transitively

Those may be leftovers from experimentation or planned extensions.

## 3. Repository Layout

### Root-level files

- `app.py`: Streamlit app entry point and role selection router
- `train.py`: offline training and model selection pipeline
- `config.yaml`: central configuration for paths and preprocessing/training constants
- `requirements.txt`: Python dependencies
- `README.md`: short project overview

### Data and artifacts

- `data/placement_cleaned.csv`: cleaned dataset used for training
- `model/best_placement_model.joblib`: serialized trained classifier
- `model/feature_columns.json`: ordered list of model input columns
- `model/model_comparison.json`: metrics for all evaluated models

### Source logic

- `src/preprocess.py`: config loading, CSV loading, encoding, train/test split, SMOTE
- `src/predict.py`: artifact loading and single-student inference
- `src/evaluate.py`: loading model comparison results
- `src/advisor.py`: Azure OpenAI client wrapper
- `src/roadmap_gen.py`: AI-powered roadmap generation and fallback logic
- `src/roadmap_pdf.py`: roadmap PDF rendering
- `src/report_gen.py`: placement report PDF rendering

### UI pages

- `views/student.py`: student-facing prediction UI
- `views/dashboard.py`: T&P dashboard UI

## 4. Configuration and Data Contracts

### 4.1 `config.yaml`

`config.yaml` defines:

- input dataset path
- output model path
- output feature column path
- output model comparison path
- model target column
- columns to drop before training
- split ratio
- random seeds
- app title and placement threshold

Current key values:

- target column: `Placement_Status`
- dropped columns: `Student_ID`, `Name`
- test size: `0.2`
- random seed: `42`
- SMOTE seed: `42`

### 4.2 Dataset schema

From `data/placement_cleaned.csv`, the raw dataset contains:

- `Student_ID`
- `Name`
- `Gender`
- `Branch`
- `10th_Percentage`
- `12th_Percentage`
- `BTech_CGPA`
- `No_of_Projects`
- `Internships`
- `Technical_Skills_Count`
- `Soft_Skills_Rating`
- `Backlogs`
- `Aptitude_Score`
- `Placement_Status`

### 4.3 Trained feature schema

After dropping identifiers and one-hot encoding categoricals with `drop_first=True`, the saved feature list is:

- `10th_Percentage`
- `12th_Percentage`
- `BTech_CGPA`
- `No_of_Projects`
- `Internships`
- `Technical_Skills_Count`
- `Soft_Skills_Rating`
- `Backlogs`
- `Aptitude_Score`
- `Gender_Male`
- `Branch_CSE`
- `Branch_ECE`
- `Branch_EEE`
- `Branch_IT`
- `Branch_ME`

Important implication:

- Because `drop_first=True` is used, one category becomes the implicit baseline.
- For `Gender`, the absence of `Gender_Male` means `Female`.
- For `Branch`, the branch not represented by a one-hot column becomes the baseline category. Given the dataset, that baseline is effectively `CE`.

This is why inference code must realign inputs to the exact saved column order.

## 5. High-Level Architecture

The project has four practical layers.

### 5.1 Presentation layer

Handled by Streamlit:

- `app.py`
- `views/student.py`
- `views/dashboard.py`

This layer manages user inputs, uploaded files, buttons, page layout, charts, and downloads.

### 5.2 ML service layer

Handled by:

- `src/preprocess.py`
- `src/predict.py`
- `src/evaluate.py`
- `train.py`

This layer knows how to:

- transform raw data into model-ready features
- train and compare models
- load the selected model
- run predictions

### 5.3 AI augmentation layer

Handled by:

- `src/advisor.py`
- `src/roadmap_gen.py`
- floating chat logic embedded in `views/student.py`

This layer is not needed for prediction itself. It adds:

- personalized roadmap generation
- an advisor chat UI using Azure OpenAI

### 5.4 Document generation layer

Handled by:

- `src/report_gen.py`
- `src/roadmap_pdf.py`

This layer converts model output and AI roadmap output into downloadable PDFs.

## 6. End-to-End Application Flow

## 6.1 App startup flow

When `streamlit run app.py` is executed:

1. `app.py` imports `views.student` and `views.dashboard`.
2. Importing those modules immediately triggers some artifact loading at module scope:
   - `views/student.py` calls `load_artifacts()` and `load_comparison_results()`
   - `views/dashboard.py` calls `load_artifacts()`
3. Streamlit page configuration is set.
4. Sidebar radio buttons determine which page’s `show()` function is executed.

Design note:

- Artifact loading currently happens at import time rather than lazily inside functions. That keeps the code simple but couples module import with filesystem availability.

## 6.2 Training flow

`train.py` runs the offline ML pipeline:

1. Load and preprocess data via `src.preprocess.load_and_preprocess()`
2. Instantiate candidate models via `get_models()`
3. Train and evaluate all models via `evaluate_all_models()`
4. Select the best model by weighted F1
5. Tune only Random Forest or XGBoost if either is the winner
6. Save:
   - trained model
   - feature column list
   - comparison metrics

This pipeline produces everything the Streamlit app needs for inference.

## 6.3 Student prediction flow

When a student uses the student page:

1. The UI collects profile information and selected skills.
2. Clicking `Predict Placement` builds a `student_dict`.
3. `predict_student()` converts that dict to a one-row DataFrame, one-hot encodes it, adds missing columns, reorders columns, and runs the classifier.
4. The page stores prediction context in `st.session_state`.
5. The page then:
   - computes SHAP values
   - displays probability and key risk factor
   - shows a gauge and radar chart
   - renders a simplified SHAP feature table
6. The student can optionally generate an AI roadmap.
7. The student can download:
   - a roadmap PDF
   - a placement report PDF
8. A floating advisor chat can be used for follow-up guidance.

## 6.4 T&P dashboard flow

The T&P page has two tabs.

### Upload & Analyze

1. User uploads a CSV.
2. The page reads the file into a DataFrame.
3. If `Placement_Status` exists, it computes summary metrics and plots.
4. If there are enough numeric columns, it renders a correlation heatmap.
5. If identifying columns exist, it shows at-risk students.

### Batch Prediction

1. User uploads a CSV with required student fields.
2. The page validates expected columns.
3. For each row, it builds a `student_dict` and calls `predict_student()`.
4. It appends prediction outputs to the DataFrame.
5. It displays summary metrics and provides a downloadable CSV.

## 7. Detailed Module-by-Module Explanation

## 7.1 `app.py`

### Purpose

This is the single Streamlit entry point. It sets global page config, draws the top title, and routes between the two dashboards.

### Behavior

- Imports `student` and `dashboard` from `views`
- Uses `st.sidebar.radio()` to let the user choose:
  - `Student Prediction`
  - `T&P Dashboard`
- Calls:
  - `student.show()` for the student workflow
  - `dashboard.show()` for the T&P workflow

### Architectural role

This file is intentionally thin. It acts as a page router, not a business-logic container.

## 7.2 `train.py`

### Purpose

This file contains the offline model training pipeline.

### Constants

- `DATA_PATH`, `MODEL_DIR`, `MODEL_PATH`, `FEATURE_PATH`, `COMPARISON_PATH`
- `TARGET_COLUMN`, `DROP_COLUMNS`, `RANDOM_STATE`

These are mostly redundant with `config.yaml`, because training actually relies on `src.preprocess.load_and_preprocess()` for dataset loading and preprocessing.

### `get_models()`

Returns a dictionary of candidate models:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- SVM
- KNN

Why this exists:

- Centralizes the available classifiers
- Makes model comparison loop-based instead of hard-coded

### `evaluate_all_models(models, X_train, X_test, y_train, y_test)`

This function:

1. Iterates through every model
2. Fits it on training data
3. Predicts labels and probabilities on the test set
4. Computes:
   - accuracy
   - precision
   - recall
   - weighted F1
   - ROC-AUC
   - 5-fold CV accuracy on the training data
5. Stores both metric rows and fitted model objects

Returns:

- `results`: serializable metric dictionaries
- `fitted_models`: trained estimator instances keyed by name

Important design choice:

- Model selection is based on weighted F1, not accuracy.
- That is reasonable when class balance or asymmetric errors matter.

### `tune_best_model(fitted_models, results, X_train, y_train)`

This function:

1. Finds the best model by highest F1 in `results`
2. Tunes only:
   - Random Forest
   - XGBoost
3. Uses `RandomizedSearchCV`
4. Returns the tuned estimator, or the original fitted estimator if tuning is skipped

Why tune only those models:

- They have richer hyperparameter spaces and often benefit more from tuning
- Simpler models are returned as-is

### `save_artifacts(model, feature_cols, results)`

Writes three output files:

- `best_placement_model.joblib`
- `feature_columns.json`
- `model_comparison.json`

This function is the bridge from offline training to online inference.

### `if __name__ == "__main__":`

This block orchestrates the full training pipeline:

1. configure logging
2. preprocess data
3. get models
4. evaluate models
5. tune best model
6. save artifacts

## 7.3 `src/preprocess.py`

### Purpose

This module owns preprocessing for training data.

### `load_config()`

Reads `config.yaml` and returns the parsed dictionary.

Why it matters:

- Keeps data paths and training constants out of the code logic
- Makes the project easier to relocate or retune

### `load_and_preprocess()`

This is the main preprocessing function. It:

1. Loads config
2. Reads the CSV dataset
3. Drops non-feature columns:
   - `Student_ID`
   - `Name`
4. Splits target from features
5. One-hot encodes categorical columns using `pd.get_dummies(..., drop_first=True)`
6. Saves the resulting feature order
7. Performs stratified train/test split
8. Applies SMOTE only on the training split
9. Returns:
   - `X_train_resampled`
   - `X_test`
   - `y_train_resampled`
   - `y_test`
   - `feature_cols`

Important design decisions:

- SMOTE is applied only to training data, which is the correct pattern.
- The model sees balanced training data but real test distribution.
- The exact feature order is preserved and later reused by inference.

## 7.4 `src/predict.py`

### Purpose

This module handles runtime inference.

### `load_artifacts()`

Loads:

- the trained classifier from `joblib`
- the saved feature column list from JSON

Returns:

- `model`
- `feature_cols`

### `predict_student(student_dict, model, feature_cols)`

This is the key inference function.

It:

1. Converts the input dict to a one-row DataFrame
2. One-hot encodes it with `drop_first=True`
3. Adds any missing columns from `feature_cols` with zero values
4. Reorders columns exactly to `feature_cols`
5. Calls:
   - `model.predict_proba(...)` for probability
   - `model.predict(...)` for class label
6. Returns:
   - `prob`: float probability of class `1`
   - `pred`: integer class label

Why this function is critical:

- The trained model expects a very specific feature order and schema.
- This alignment step prevents runtime shape mismatches.

## 7.5 `src/evaluate.py`

### Purpose

This module reads saved comparison results for display.

### `load_comparison_results()`

Reads `model/model_comparison.json`, converts it into a DataFrame, rounds metric columns, and returns it.

Used by:

- `views/student.py`, to show the model comparison table

### `get_best_model_name(df)`

Returns the model name with the highest F1 score.

This helper is not currently used in the UI, because `views/student.py` computes the best row directly.

## 7.6 `src/advisor.py`

### Purpose

This module wraps Azure OpenAI access.

### `get_azure_client()`

Loads environment variables from `.env` and initializes an `AzureOpenAI` client.

Expected environment variables:

- `AZURE_OPENAI_MINI_ENDPOINT`
- `AZURE_OPENAI_MINI_API_KEY`
- `AZURE_OPENAI_MINI_API_VERSION`

If credentials are missing or invalid, it raises a `ValueError`.

### `get_deployment_name()`

Reads and returns:

- `AZURE_OPENAI_MINI_DEPLOYMENT`

### `chat_complete(messages, temperature=0.7, max_completion_tokens=1000)`

Calls Azure OpenAI chat completions and returns the response text as a string.

Used by:

- `src/roadmap_gen.py`

Important note:

- This module is the backend AI wrapper.
- The floating chat widget in `views/student.py` does not use this wrapper. It calls the Azure endpoint directly from embedded JavaScript.

## 7.7 `src/roadmap_gen.py`

### Purpose

This module generates a personalized placement roadmap using Azure OpenAI.

It combines:

- student profile fields
- SHAP factors
- selected skills
- a field-detection heuristic
- pre-authored resource maps

### `FIELD_RESOURCES`

A large static dictionary mapping career domains to curated resources:

- learning paths
- practice resources
- company targets
- certifications
- GitHub project ideas
- interview preparation topics

This acts like a lightweight knowledge base.

### `_build_user_prompt(...)`

Builds a prompt from student data and SHAP factors.

Important note:

- This helper is currently not used by `generate_roadmap()`.
- It looks like an earlier or alternate prompt-construction approach left in the file.

### `_fallback_roadmap()`

Returns a safe default roadmap object if generation or parsing fails.

Why it matters:

- Prevents the UI from crashing if the model returns malformed output
- Ensures the page always receives a dict with expected keys

### `detect_field(skills)`

Infers the student’s likely target domain from selected skills using keyword scoring.

Possible outputs include:

- `AI/ML & Data Science`
- `Web Development (Full Stack)`
- `Cloud & DevOps`
- `Cybersecurity`
- `Software Engineering (General)`
- `Mobile Development`
- `Data Engineering & Databases`
- `Embedded & IoT`
- `Mechanical/Civil CAD`
- `Competitive Programming & Product`

If no match is found, it defaults to `Software Engineering (General)`.

### `extract_json(text)`

Attempts to parse a model response as JSON.

It can handle:

- raw JSON
- fenced markdown code blocks
- JSON embedded in a larger response body

This is a defensive parser for LLM variability.

### `generate_roadmap(student_dict, shap_factors, prediction_prob, skills=None)`

This is the main roadmap generator.

It:

1. Detects a target field from skills
2. Looks up field-specific resources
3. Builds a strongly constrained prompt demanding JSON-only output
4. Calls `chat_complete()`
5. Parses the response with `extract_json()`
6. Retries once with a stricter prompt if parsing fails
7. Falls back to `_fallback_roadmap()` if both attempts fail

Returned roadmap structure includes:

- `detected_field`
- `summary`
- `probability_context`
- `phases`
- `quick_wins`
- `companies_to_target`
- `skills_to_learn`
- `certifications`
- `project_ideas`
- `interview_prep`

Architectural significance:

- This file is where the application moves from ML prediction into advisory intelligence.
- The roadmap is not derived mathematically from the classifier alone; it is synthesized from model signals plus curated domain knowledge plus LLM generation.

## 7.8 `src/roadmap_pdf.py`

### Purpose

This module renders the roadmap into a structured PDF.

### `sanitize(text)`

Converts unsupported Unicode punctuation and symbols into Latin-1-safe equivalents so PDF generation does not fail.

### `RoadmapPDF(FPDF)`

Custom PDF class with:

- empty header
- custom footer

### Layout helper functions

- `add_page_title(pdf, title, subtitle="")`
- `add_section_heading(pdf, text)`
- `add_body_text(pdf, text)`
- `add_label_value(pdf, label, value)`
- `add_bullet_list(pdf, items)`
- `add_numbered_list(pdf, items)`
- `add_phase_block(pdf, phase)`

These functions keep PDF drawing logic modular and readable.

### `generate_roadmap_pdf(student_name, roadmap)`

Builds the final roadmap PDF.

Sections may include:

- title
- assessment
- probability context
- 6-month action plan
- quick wins
- companies to target
- skills to learn
- certifications
- project ideas
- interview prep

Returns raw PDF bytes for Streamlit download.

## 7.9 `src/report_gen.py`

### Purpose

This module generates a prediction report PDF for a student.

### `REPORT_FEATURE_LABELS`

Maps raw feature names into student-friendly display labels for SHAP presentation.

### `_latin1_safe(text)`

Converts text into Latin-1-safe output for PDF compatibility.

### `PlacementReport(FPDF)`

Custom PDF class with:

- centered header
- footer showing page number and generation date

### `generate_report(student_dict, prob, pred, shap_df)`

Creates the PDF report with:

- prediction result banner
- probability
- student academic profile
- SHAP factor table
- personalized rule-based recommendations

This function combines three signal sources:

- original student inputs
- ML prediction outputs
- SHAP explanation outputs

It also contains deterministic recommendation rules such as:

- warn if backlogs > 0
- warn if projects < 2
- warn if CGPA < 7
- warn if skills < 5
- recommend internship if none exists

## 7.10 `views/student.py`

### Purpose

This is the most feature-rich file in the project. It implements the entire student-facing experience.

### Module-level initialization

At import time, this file:

- loads the trained model and feature columns via `load_artifacts()`
- loads model comparison metrics via `load_comparison_results()`

### `FEATURE_LABELS`

Maps internal feature names to human-readable display labels for SHAP explanations.

### `SKILL_OPTIONS`

A curated list of selectable skills used by the student form.

### `_escape_html(value)`

HTML-escapes text for safe insertion into custom Streamlit-rendered markup.

### `_render_shap_inline_table(shap_df)`

Builds a compact custom SHAP explanation table using inline HTML and CSS.

Behavior:

- takes top SHAP factors
- scales bar widths by absolute contribution
- uses green/red coloring for positive/negative impact

### `render_floating_chat(student_dict, prob, skills, pred)`

This function injects a full floating chat widget through `streamlit.components.v1.html`.

What it does:

- reads Azure OpenAI variables from `.env`
- builds a system prompt using the student’s profile
- generates a full HTML/CSS/JS widget
- sends chat requests directly from JavaScript to the Azure OpenAI REST endpoint

Important architectural detail:

- This is a client-side AI integration embedded inside the Streamlit page.
- It bypasses `src/advisor.py` and calls Azure directly from the front-end component.
- That makes the chat UI self-contained but also couples the page to browser-side API access.

### `show()`

This is the main student page renderer.

Its responsibilities are broad:

1. Inject page-level CSS customizations
2. Render the page heading
3. Show model comparison metrics in an expander
4. Collect student profile inputs
5. Manage skill selection and custom skills
6. Trigger prediction
7. Persist results in `st.session_state`
8. Reconstruct aligned input data for SHAP
9. Build SHAP explanations using `shap.KernelExplainer`
10. Render:
    - metrics
    - probability gauge
    - radar chart
    - SHAP explanation
11. Generate AI roadmap on demand
12. Render roadmap sections
13. Generate roadmap PDF
14. Generate placement report PDF
15. Show deterministic recommendations
16. Render the floating advisor chat

### Student page state model

The page uses `st.session_state` for:

- `selected_skills`
- `student_skills`
- `latest_prediction`
- `roadmap_result`

This avoids losing prediction context across reruns.

### Student page output layers

The student experience combines four different engines:

1. deterministic form input collection
2. trained ML classification
3. SHAP explanation
4. AI-based advisory generation

This is the core product experience of the application.

## 7.11 `views/dashboard.py`

### Purpose

This file implements the Training and Placement dashboard.

### Module-level initialization

At import time, it loads:

- `model`
- `feature_cols`

It also imports `load_comparison_results` but does not use it.

### `show()`

This single function handles the full T&P experience.

It creates two tabs:

- `Upload & Analyze`
- `Batch Prediction`

### Upload & Analyze tab

Capabilities:

- upload a CSV
- preview the data
- compute total students, placed, not placed, placement rate
- show placement ratio pie chart
- show placement-by-branch histogram
- show CGPA and project distributions by placement
- render numeric correlation heatmap
- list at-risk students when identifiers are present

This tab is descriptive analytics over user-uploaded data.

### Batch Prediction tab

Capabilities:

- validate uploaded CSV columns
- iterate over every student row
- call `predict_student()` for each student
- store probability and binary result
- show summary metrics
- allow CSV download

This tab is predictive analytics over uploaded data.

### Error handling

The whole page body is wrapped in a `try/except` block. If something breaks, the page shows:

- the error string
- the traceback

That is useful during development, though it exposes internal details to end users.

## 8. Saved Artifacts and Their Role

## 8.1 `model/best_placement_model.joblib`

This is the serialized classifier chosen by the training pipeline.

The app assumes:

- it exists
- it is compatible with `predict()` and `predict_proba()`

## 8.2 `model/feature_columns.json`

This is the most important artifact besides the model itself.

Why:

- it defines the exact inference schema
- runtime input alignment depends on it
- training and inference stay synchronized through this file

## 8.3 `model/model_comparison.json`

Stores evaluation metrics for all candidate models.

Current saved results show:

- best weighted F1: Logistic Regression
- best ROC-AUC: Random Forest

This means the selected “best” model depends on the chosen metric. The code uses weighted F1, so Logistic Regression wins in the saved artifact set.

## 9. Runtime Data Flow

## 9.1 Offline training data flow

`placement_cleaned.csv`

-> drop identifier columns

-> split `X` and `y`

-> one-hot encode categorical fields

-> save feature order

-> split train/test

-> apply SMOTE to training only

-> train candidate models

-> compute evaluation metrics

-> choose/tune winner

-> save artifacts

## 9.2 Online student prediction data flow

Student form inputs

-> `student_dict`

-> one-row DataFrame

-> one-hot encode

-> add missing training columns

-> reorder to `feature_columns.json`

-> `predict()` and `predict_proba()`

-> probability + label

-> SHAP explanation

-> charts + roadmap + PDFs

## 9.3 Online batch prediction data flow

Uploaded CSV

-> iterate rows

-> build `student_dict` per row

-> `predict_student()`

-> collect probabilities and labels

-> append results to DataFrame

-> display and export CSV

## 10. UI and UX Design Choices

### Student page

The student page is intentionally richer than the dashboard.

It includes:

- custom CSS for visual polish
- a compact prediction journey
- explainability output
- personalized recommendations
- AI roadmap generation
- AI advisor chat

This makes it feel more like a student product experience than a plain analytics form.

### T&P page

The T&P dashboard is intentionally simpler:

- upload-first workflow
- descriptive charts
- practical at-risk table
- batch prediction export

Its target user is an administrator or placement coordinator, not a student.

## 11. Function Inventory Summary

For quick reference, here is the functional surface of the codebase.

### Training and preprocessing

- `train.get_models()`
- `train.evaluate_all_models(...)`
- `train.tune_best_model(...)`
- `train.save_artifacts(...)`
- `src.preprocess.load_config()`
- `src.preprocess.load_and_preprocess()`

### Inference and evaluation

- `src.predict.load_artifacts()`
- `src.predict.predict_student(...)`
- `src.evaluate.load_comparison_results()`
- `src.evaluate.get_best_model_name(df)`

### AI integration

- `src.advisor.get_azure_client()`
- `src.advisor.get_deployment_name()`
- `src.advisor.chat_complete(...)`
- `src.roadmap_gen.detect_field(skills)`
- `src.roadmap_gen.extract_json(text)`
- `src.roadmap_gen.generate_roadmap(...)`

### PDF generation

- `src.roadmap_pdf.generate_roadmap_pdf(...)`
- `src.report_gen.generate_report(...)`

### UI helpers

- `views.student._escape_html(...)`
- `views.student._render_shap_inline_table(...)`
- `views.student.render_floating_chat(...)`
- `views.student.show()`
- `views.dashboard.show()`

## 12. Architectural Strengths

This codebase has several solid qualities.

### Clear responsibility split

Training, inference, UI, AI advisory, and PDF generation are separated into different files.

### Practical artifact strategy

Persisting both the model and the feature schema is the right approach for tabular ML deployment.

### Defensive LLM integration

Roadmap generation includes retry and fallback logic, which is important in production-like workflows.

### Useful explainability layer

SHAP gives the prediction page much more credibility and educational value.

### Dual-audience design

The project successfully serves two user types:

- individual students
- T&P administrators

## 13. Architectural Limitations and Observations

This section is explanatory, not a code review. These points help understand the current design tradeoffs.

### Import-time artifact loading

Both UI modules load model artifacts at module import time. That is simple, but it means:

- startup depends on artifact files existing immediately
- tests and imports are more tightly coupled to the filesystem

### Prompt helper duplication

`src/roadmap_gen._build_user_prompt()` exists but is not used by `generate_roadmap()`.

### Front-end and back-end AI split

There are two different Azure chat access patterns:

- backend wrapper in `src/advisor.py`
- direct JavaScript fetch inside `views/student.py`

This means AI integration is not fully centralized.

### SHAP cost and latency

The student page computes SHAP explanations at runtime using `KernelExplainer`, which is model-agnostic but slower than model-specific explainers.

### Requirements breadth

`requirements.txt` includes packages that the current code paths do not use, suggesting the environment is broader than the active implementation.

### Config duplication

Some paths/constants appear both in `config.yaml` and `train.py`, although actual preprocessing relies on the config file.

## 14. How the Pieces Fit Together Conceptually

This project is not just “a classifier with a UI.” It is really a four-part system:

1. A tabular ML engine predicts placement outcomes.
2. An explainability engine shows why that prediction happened.
3. An advisory layer converts model signals into human guidance.
4. A reporting/export layer turns outcomes into shareable documents.

That combination is what makes the project feel like a product rather than a notebook demo.

## 15. Suggested Mental Model for Understanding the Project

If you need to explain this codebase in an interview, viva, or documentation handoff, the simplest mental model is:

- `train.py` builds the brain.
- `model/` stores the brain and its expected input shape.
- `src/predict.py` is the inference adapter.
- `views/student.py` is the student-facing application.
- `views/dashboard.py` is the administrator-facing application.
- `src/roadmap_gen.py` and `src/advisor.py` add AI guidance.
- `src/report_gen.py` and `src/roadmap_pdf.py` turn results into downloadable artifacts.

## 16. Fast Walkthrough for a New Developer

If a new developer joins the project, the best reading order is:

1. `app.py`
2. `views/student.py`
3. `views/dashboard.py`
4. `src/predict.py`
5. `src/preprocess.py`
6. `train.py`
7. `src/roadmap_gen.py`
8. `src/report_gen.py`
9. `src/roadmap_pdf.py`
10. `config.yaml`

That order starts with the user-visible behavior and then works backward into the model and support layers.

## 17. Final Conclusion

This codebase is a compact full-stack ML application centered on student placement prediction. Its architecture is straightforward and pragmatic:

- train offline
- save artifacts
- load artifacts in Streamlit
- infer on single or batch data
- explain results
- extend them with AI-generated guidance
- export documents

The strongest idea in the system is the combination of prediction, explainability, and actionability. The model does not just output a label. The application also tries to answer:

- why the label happened
- what the student should do next
- how T&P teams can act on the data at scale

That makes the project useful for both demonstration and real academic/product discussion.
