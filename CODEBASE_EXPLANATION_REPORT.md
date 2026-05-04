# Placement Prediction System - Complete Codebase Explanation Report

## 1. Executive Summary

This repository implements a small end-to-end machine learning product for engineering college placement prediction.

It has two major operating modes:

1. Offline training mode
   - Reads a cleaned CSV dataset
   - Encodes categorical features
   - Splits train and test data
   - Applies SMOTE on the training split
   - Evaluates multiple classifiers
   - Selects the best model by weighted F1 score
   - Saves the trained model and metadata artifacts

2. Online application mode
   - Loads the saved model and feature schema
   - Accepts a single student's profile through a Streamlit UI
   - Predicts placement probability and binary outcome
   - Computes a local SHAP explanation
   - Generates a human-readable PDF report
   - Optionally generates an AI roadmap and roadmap PDF
   - Offers a T&P dashboard for analytics and batch predictions

At the architectural level, the codebase is split into:

- `app.py` for Streamlit bootstrapping and page routing
- `views/` for UI pages
- `src/` for reusable ML, AI, and PDF logic
- `train.py` for artifact generation
- `data/` for training data
- `model/` for persisted artifacts

The project is not a layered enterprise application with database, API server, and background workers. It is a direct desktop-style Streamlit application whose business logic is imported in-process. That simplicity makes the runtime easy to follow.

## 2. Repository Inventory

### Root files

- `app.py`
  - Streamlit entry point
  - Page setup
  - Role selection
  - Floating chat render trigger

- `train.py`
  - Offline model training pipeline
  - Model comparison
  - Optional tuning for selected models
  - Artifact persistence

- `config.yaml`
  - Paths and training configuration

- `requirements.txt`
  - Python dependency lock-style list

- `README.md`
  - Short human-facing overview

- `CODEBASE_EXPLANATION_REPORT.md`
  - This report

### Data and artifacts

- `data/placement_cleaned.csv`
  - Cleaned supervised learning dataset

- `model/best_placement_model.joblib`
  - Serialized trained classifier

- `model/feature_columns.json`
  - Ordered model input schema after one-hot encoding

- `model/model_comparison.json`
  - Stored metrics for all evaluated models

### Python packages

- `src/`
  - Core non-UI logic

- `views/`
  - Streamlit pages

- `__pycache__/`
  - Python bytecode cache

### Notebook folders

- `notebooks/`
- `model/Untitled.ipynb`

These do not participate in the main production flow shown by the application.

## 3. Actual Runtime Architecture

The architecture is best understood as two connected pipelines.

### 3.1 Training pipeline

`data/placement_cleaned.csv`
-> `src/preprocess.load_and_preprocess()`
-> encoded feature matrix + split data
-> `train.py` model evaluation loop
-> best model selection
-> optional hyperparameter tuning
-> save to `model/`

### 3.2 Application pipeline

User input in Streamlit
-> `views/student.show()`
-> `src.predict.predict_student()`
-> prediction probability + label
-> SHAP explanation inside `views/student.show()`
-> optional roadmap generation via `src.roadmap_gen.generate_roadmap()`
-> optional PDFs via `src.report_gen.generate_report()` and `src.roadmap_pdf.generate_roadmap_pdf()`

### 3.3 Dashboard pipeline

Uploaded CSV
-> `views/dashboard.show()`
-> analytics charts and metrics
or
-> row-by-row calls into `src.predict.predict_student()`
-> downloadable batch prediction CSV

## 4. Current Dataset and Artifact Facts

These are not assumptions. They come from the files currently in the repository.

### Dataset shape

- Rows: `200`
- Columns: `14`

### Dataset columns

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

### Target distribution

- `Placement_Status = 1`: `142`
- `Placement_Status = 0`: `58`

This is why SMOTE is used during training: the classes are imbalanced.

### Branch distribution

- `CSE`: `38`
- `ME`: `37`
- `CE`: `34`
- `EEE`: `32`
- `IT`: `30`
- `ECE`: `29`

### Gender distribution

- `Male`: `124`
- `Female`: `76`

### Saved feature schema

The trained model expects exactly these `15` input features:

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

- `Gender_Female` is the dropped baseline category
- `Branch_CE` is the dropped baseline category because `pd.get_dummies(..., drop_first=True)` was used

### Saved active model

The persisted model in `model/best_placement_model.joblib` is currently:

- `LogisticRegression(max_iter=1000, random_state=42)`

### Stored comparison results

The saved comparison file shows:

- Logistic Regression: best weighted F1
- Random Forest: tied or comparable accuracy, higher ROC-AUC than logistic regression
- XGBoost, Gradient Boosting, SVM, KNN also evaluated

The application currently treats Logistic Regression as the deployed model because that is the object saved in `best_placement_model.joblib`.

## 5. File-by-File Explanation

## 5.1 `app.py`

Purpose:

- Initializes Streamlit
- Injects global CSS
- Routes between student and dashboard pages
- Renders the floating AI chat only after a student prediction exists

How it works:

1. `st.set_page_config(...)` sets page title and wide layout.
2. A large CSS block customizes top padding and fixes the custom component iframe so the chatbot behaves like a floating overlay.
3. A styled title is rendered via `st.markdown(...)`.
4. A sidebar radio decides between:
   - `Student Prediction`
   - `T&P Dashboard`
5. Depending on selection:
   - `student.show()` is called
   - or `dashboard.show()` is called
6. After the student page renders, if `st.session_state["latest_prediction"]` exists, `student.render_floating_chat(...)` is called with the latest prediction context.

Architectural role:

- This is the root composition layer.
- It contains almost no business logic.
- It wires pages together and coordinates cross-page UI behavior.

## 5.2 `train.py`

Purpose:

- Trains candidate models
- Evaluates them
- Selects the best one
- Optionally tunes it
- Saves output artifacts

### Key constants

The file defines model paths and some local constants:

- `DATA_PATH`
- `MODEL_DIR`
- `MODEL_PATH`
- `FEATURE_PATH`
- `COMPARISON_PATH`

However, note that training does not actually use `DATA_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`, or `RANDOM_STATE` directly for preprocessing. It delegates to `src.preprocess.load_and_preprocess()`, which reads `config.yaml`. So some constants in `train.py` are redundant.

### `get_models()`

Returns six model instances:

- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting
- SVM with probabilities enabled
- KNN

This function is the model registry for the experiment.

### `evaluate_all_models(...)`

Responsibilities:

- Fit each candidate model
- Predict on the test set
- Compute metrics
- Compute cross-validation accuracy on training data
- Collect fitted models and result metadata

For each model:

1. `.fit(X_train, y_train)`
2. `predict(X_test)` for class labels
3. `predict_proba(X_test)[:, 1]` for placement probabilities
4. Compute:
   - accuracy
   - weighted precision
   - weighted recall
   - weighted F1
   - ROC-AUC
   - 5-fold CV accuracy mean and std
5. Print classification report
6. Store the fitted model in a dictionary

Output:

- `results`: list of metric dictionaries
- `fitted_models`: mapping from model name to trained estimator

### `tune_best_model(...)`

Responsibilities:

- Select the best model by highest weighted F1
- Tune it only if it is Random Forest or XGBoost

Logic:

1. Find the max of `results` using `f1`
2. If best model is:
   - `Random Forest`: define RF hyperparameter grid
   - `XGBoost`: define XGBoost hyperparameter grid
   - anything else: skip tuning and return the fitted model directly
3. Run `RandomizedSearchCV` with:
   - `n_iter=20`
   - `cv=5`
   - `scoring="f1"`
   - `n_jobs=-1`
4. Return `search.best_estimator_`

Current behavior in this repository:

- Logistic Regression has the highest saved weighted F1
- Therefore tuning is skipped
- The already fitted Logistic Regression model is persisted as the production model

### `save_artifacts(...)`

Writes three files:

- joblib model
- feature column JSON
- model comparison JSON

This is the key boundary between offline training and online inference.

### `if __name__ == "__main__":`

Execution order:

1. Configure logging
2. Load and preprocess data
3. Build candidate models
4. Evaluate all models
5. Tune the best model if supported
6. Save artifacts

## 5.3 `src/preprocess.py`

Purpose:

- Centralized training-time preprocessing

### `load_config()`

- Reads `config.yaml`
- Returns parsed YAML as Python dict

### `load_and_preprocess()`

This is the training data preparation pipeline.

Step-by-step:

1. Load config
2. Resolve `data_path`
3. Read CSV into a DataFrame
4. Drop configured identifier columns:
   - `Student_ID`
   - `Name`
5. Split into:
   - `X`: feature columns
   - `y`: `Placement_Status`
6. Apply one-hot encoding using `pd.get_dummies(X, drop_first=True)`
7. Store the resulting ordered feature column names
8. Split into train and test with:
   - `test_size = 0.2`
   - `random_state = 42`
   - `stratify = y`
9. Apply SMOTE only on the training split
10. Return:
   - `X_train_resampled`
   - `X_test`
   - `y_train_resampled`
   - `y_test`
   - `feature_cols`

Important architectural detail:

- The preprocessing logic used in training is not packaged as a scikit-learn pipeline object.
- Instead, training and inference manually repeat encoding logic.
- The feature schema is synchronized by saving `feature_columns.json`.

## 5.4 `src/predict.py`

Purpose:

- Runtime artifact loading
- Single-student inference

### `load_artifacts()`

Loads:

- `best_placement_model.joblib`
- `feature_columns.json`

Returns:

- `model`
- `feature_cols`

### `predict_student(student_dict, model, feature_cols)`

This is the inference adapter between UI input and the trained model.

Step-by-step:

1. Wrap the input dictionary in a one-row DataFrame
2. Apply `pd.get_dummies(..., drop_first=True)` to match training style
3. For every expected trained column missing in the input row, add it with zero
4. Reorder columns to exactly match `feature_cols`
5. Run:
   - `model.predict_proba(...)` to get probability
   - `model.predict(...)` to get binary class
6. Return `(prob, pred)`

Why this works:

- Since training saved the exact feature order, inference can reconstruct the same vector shape.
- Baseline categories are represented implicitly by missing dummy columns that are then filled with zero.

## 5.5 `src/evaluate.py`

Purpose:

- Load saved model comparison metrics for UI display

### `load_comparison_results()`

1. Read `model/model_comparison.json`
2. Convert it to a DataFrame
3. Round numeric columns to 4 decimals
4. Return the DataFrame

### `get_best_model_name(df)`

- Finds the row with max `f1`
- Returns its `model_name`

This module is used by the student page to explain which model family was selected during training.

## 5.6 `src/advisor.py`

Purpose:

- Azure OpenAI wrapper for server-side roadmap generation

### How it works

1. `load_dotenv(...)` loads `.env`
2. `get_azure_client()` reads:
   - `AZURE_OPENAI_MINI_ENDPOINT`
   - `AZURE_OPENAI_MINI_API_KEY`
   - `AZURE_OPENAI_MINI_API_VERSION`
3. If any are missing, it raises a `ValueError`
4. It instantiates `openai.AzureOpenAI`
5. `get_deployment_name()` reads `AZURE_OPENAI_MINI_DEPLOYMENT`
6. `chat_complete(...)` sends a chat completion request and returns the response text

Architectural role:

- This is the server-side LLM integration used by `src.roadmap_gen`.

## 5.7 `src/roadmap_gen.py`

Purpose:

- Turn a student's profile plus SHAP factors plus skills into a structured placement roadmap

This module is more than a thin LLM wrapper. It contains domain logic before prompting the model.

### `FIELD_RESOURCES`

This is a large static knowledge base mapping target placement fields to:

- learning resources
- practice ideas
- target companies
- certifications
- GitHub project ideas
- interview preparation topics

Supported fields include:

- AI/ML & Data Science
- Web Development (Full Stack)
- Cloud & DevOps
- Cybersecurity
- Software Engineering (General)
- Mobile Development
- Data Engineering & Databases
- Embedded & IoT
- Mechanical/Civil CAD
- Competitive Programming & Product

### `_fallback_roadmap()`

Returns a minimal empty roadmap when AI generation fails.

This is important because the UI can still continue rendering without crashing.

### `detect_field(skills)`

This function infers the student's likely target placement field using keyword matching over the selected skills list.

How it works:

1. Lowercase all skills
2. For each field, count keyword hits
3. Return the field with the highest score
4. If nothing matches, return `Software Engineering (General)`

This is deterministic heuristic classification, not ML.

### `extract_json(text)`

Robustness helper:

- Tries to parse JSON directly
- If the model wrapped JSON in code fences or extra text, it strips fences or extracts the first JSON-looking object with regex

### `generate_roadmap(...)`

This is the main roadmap generation flow.

Inputs:

- `student_dict`
- `shap_factors`
- `prediction_prob`
- optional `skills`

Execution flow:

1. Normalize `skills`
2. Infer target field via `detect_field`
3. Pull field-specific resource bundle from `FIELD_RESOURCES`
4. Build a compact SHAP summary using top five factors
5. Build a system prompt describing the advisor persona
6. Build a long user prompt containing:
   - academic details
   - profile metrics
   - target field
   - placement probability
   - SHAP factors
   - field-specific resource names
   - exact JSON schema to return
7. Send the request through `chat_complete(...)`
8. Parse the response as JSON
9. If parsing fails, retry once with a stricter message
10. If it still fails, return the fallback roadmap

Output shape:

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

Architectural role:

- This module bridges ML explainability and generative AI guidance.
- It does not affect the prediction result itself.
- It is a post-prediction advisory subsystem.

## 5.8 `src/report_gen.py`

Purpose:

- Generate a classic PDF placement report for a single student

### Main pieces

- `REPORT_FEATURE_LABELS`
  - Human-readable labels for technical feature names

- `_latin1_safe(text)`
  - Sanitizes strings for FPDF compatibility

- `PlacementReport(FPDF)`
  - Custom header and footer

- `generate_report(student_dict, prob, pred, shap_df)`
  - Builds the full PDF

### PDF structure

1. Header
   - App title
   - Report title

2. Prediction Result section
   - Binary outcome banner
   - Probability text

3. Student Academic Profile section
   - Tabular summary of major inputs

4. SHAP Analysis section
   - Feature
   - SHAP value
   - Positive/Negative impact

5. Personalized Recommendations section
   - Rule-based recommendation bullets based on profile weaknesses

Important point:

- These recommendations are deterministic heuristics, not generated by the model or by AI.

## 5.9 `src/roadmap_pdf.py`

Purpose:

- Generate a visually richer PDF for the AI roadmap

This file is presentation-heavy. Its job is layout, color, typography, spacing, and section rendering.

### Structure

- Color palette constants
- Text sanitation helper `_s(...)`
- `RoadmapPDF(FPDF)` subclass with custom footer
- Multiple layout helpers:
  - `_draw_header`
  - `_section_title`
  - `_bullet`
  - `_numbered`
  - `_phase_card`
  - `_chips`
  - `_info_box`
  - `_two_col_list`
- `generate_roadmap_pdf(student_name, roadmap)`

### Output sections

- Header banner
- Placement assessment
- Probability context
- 6-month action plan
- Quick wins
- Companies to target
- Skills to build
- Certifications
- Project ideas
- Interview prep areas

Architectural role:

- Pure formatting layer for the roadmap JSON generated by `src/roadmap_gen`.

## 5.10 `views/student.py`

Purpose:

- Entire student-facing application workflow

This is the most complex file in the repository because it contains:

- form UI
- prediction trigger
- SHAP explanation generation
- metrics and charts
- AI roadmap UI
- PDF download buttons
- rule-based recommendations
- floating AI chatbot implementation

### Module-level initialization

At import time it executes:

- `model, feature_cols = load_artifacts()`
- `comparison_df = load_comparison_results()`

This means the model is loaded once when the page module is imported rather than on each prediction click.

### Helper data

- `FEATURE_LABELS`
  - Human-readable display labels for SHAP features

- `SKILL_OPTIONS`
  - Long hard-coded selectable skill list used by the roadmap and chatbot context

### `_render_shap_inline_table(shap_df)`

This helper converts SHAP output into a styled mini horizontal bar display rendered via HTML and CSS.

### `render_floating_chat(...)`

This is a self-contained chatbot rendered as a Streamlit HTML component.

Its behavior:

1. Reads Azure credentials directly from `.env`
2. Builds a student-aware system prompt containing:
   - branch
   - CGPA
   - 10th/12th scores
   - projects
   - internships
   - backlogs
   - selected skills
   - soft skills
   - aptitude
   - prediction result
   - probability
3. Constructs a large HTML string with:
   - floating button
   - chat panel
   - message area
   - text box
   - JavaScript formatting helpers
4. Creates a browser-side `fetch(...)` call to the Azure OpenAI endpoint
5. Maintains client-side conversation history in JavaScript

Architectural note:

- This chatbot does not go through the Python `src/advisor.py` wrapper.
- It calls Azure OpenAI directly from the browser.

### `show()`

This is the core student page renderer.

#### Step A: Page styling and header

- Injects CSS
- Renders page title and subtitle

#### Step B: Model info section

- Opens an expander describing the trained model comparison
- Displays metrics table from `comparison_df`
- Highlights the best model by F1

#### Step C: Input form

The page is split into 3 columns:

- Column 1:
  - 10th percentage
  - 12th percentage
  - CGPA
  - projects

- Column 2:
  - internships
  - multi-select skills
  - custom skill input
  - derived technical skill count
  - soft skills slider
  - aptitude slider

- Column 3:
  - backlogs
  - gender
  - branch
  - predict button

Important implementation detail:

- `Technical_Skills_Count` is derived from the number of selected plus custom skills
- If no skills are selected, it falls back to `1` rather than `0`

#### Step D: Prediction trigger

When the user clicks `Predict Placement`:

1. Old roadmap state is cleared
2. Selected skills are stored in `st.session_state["student_skills"]`
3. A `student_dict` is assembled
4. `predict_student(...)` is called
5. The result is saved into `st.session_state["latest_prediction"]`

Why session state is used:

- Streamlit reruns the script on interaction
- Session state preserves the latest prediction across reruns
- That allows the charts, PDF buttons, and chat widget to remain available

#### Step E: SHAP generation

If `latest_prediction` exists:

1. Rebuild the student input as one-hot encoded `sdf`
2. Re-run `load_and_preprocess()` to get training data
3. Use the training data as SHAP background
4. Construct `shap.KernelExplainer(model.predict_proba, shap.sample(background, 50))`
5. Compute `shap_values` with `nsamples=100`
6. Normalize the output depending on SHAP return shape
7. Build `shap_df`
8. Sort features by absolute SHAP magnitude
9. Keep top 10
10. Add positive/negative labels

This SHAP logic is local explanation only for the current prediction.

#### Step F: Prediction result display

Renders:

- metric cards
  - placement result
  - probability
  - key risk factor

- gauge chart
  - probability and threshold at 50%

- radar chart
  - 10th
  - 12th
  - CGPA
  - projects
  - technical skills
  - soft skills
  - aptitude

- SHAP explanation table

#### Step G: AI roadmap generation

On `Generate My Placement Roadmap`:

1. Convert `shap_df` into a simpler list of dictionaries
2. Call `generate_roadmap(...)`
3. Store roadmap JSON in session state
4. Render:
   - summary
   - probability context
   - up to 3 phase cards
   - quick wins
   - companies
   - skills to learn
5. Generate roadmap PDF and expose a download button

There is also backward compatibility logic:

- If `generate_roadmap(...)` does not accept the `skills` keyword, the code retries without it

That suggests the function signature changed at some point.

#### Step H: Report PDF

- Calls `src.report_gen.generate_report(...)`
- Exposes a download button

#### Step I: Rule-based recommendations

After the prediction, the page shows simple recommendations based on thresholds:

- backlogs > 0
- projects < 2
- CGPA < 7.0
- technical skills < 5
- internships == 0
- aptitude < 6
- strong profile case

These are deterministic and separate from the AI roadmap.

## 5.11 `views/dashboard.py`

Purpose:

- T&P analytics page
- Batch prediction page

### Module-level initialization

- `model, feature_cols = load_artifacts()`

Again, this loads the model once at import time.

### `show()`

The function wraps the whole page in a `try/except`, so UI errors are shown on-screen with traceback.

#### Tab 1: Upload & Analyze

Workflow:

1. User uploads a CSV
2. DataFrame is loaded and previewed
3. If `Placement_Status` exists, the page computes:
   - total students
   - placed count
   - not placed count
   - placement rate
4. It renders:
   - pie chart for placement ratio
   - branch histogram if branch exists
5. If `BTech_CGPA` and `Placement_Status` exist:
   - box plot by placement status
6. If `No_of_Projects` and `Placement_Status` exist:
   - projects box plot by placement status
7. If enough numeric columns exist:
   - correlation heatmap
8. If identifiers and `Placement_Status` exist:
   - at-risk table for non-placed students

Important point:

- This tab is descriptive analytics.
- It does not use the ML model unless the uploaded dataset already contains `Placement_Status`.
- It analyzes the uploaded data as-is.

#### Tab 2: Batch Prediction

Workflow:

1. User uploads a CSV
2. Preview is shown
3. User clicks `Run Batch Prediction`
4. File is validated against required columns
5. For each row:
   - build `student_dict`
   - call `predict_student(...)`
   - append probability and prediction
   - update progress bar
6. Final result adds:
   - `Placement_Prediction`
   - `Placement_Probability_%`
   - `Result`
7. Summary metrics are rendered
8. Full result table is shown
9. User can download predictions as CSV

Architectural note:

- Batch inference is implemented as a Python loop over rows
- It is simple and easy to understand
- It is not vectorized

## 5.12 `views/__init__.py` and `src/__init__.py`

These are package markers.

They do not add behavior.

## 6. End-to-End Flow Explanations

## 6.1 Full training flow

1. Developer runs:
   - `python train.py`
2. `train.py` calls `load_and_preprocess()`
3. Preprocessing:
   - reads config
   - loads CSV
   - drops `Student_ID` and `Name`
   - one-hot encodes `Gender` and `Branch`
   - train/test split
   - SMOTE on training data only
4. Six candidate models are trained and evaluated
5. The best model is selected by weighted F1
6. If the best model is RF or XGBoost, it is tuned with `RandomizedSearchCV`
7. Artifacts are saved in `model/`
8. The Streamlit app can now use those artifacts

## 6.2 Full student prediction flow

1. User opens app with:
   - `streamlit run app.py`
2. `app.py` imports `views.student`
3. `views.student` loads model and comparison artifacts
4. User enters profile data and clicks `Predict Placement`
5. A `student_dict` is assembled
6. `predict_student()`:
   - one-hot encodes the single row
   - aligns it to saved training columns
   - runs model prediction
7. Result is stored in session state
8. Same page reruns and sees `latest_prediction`
9. SHAP explanation is computed for the current row
10. UI renders:
   - metrics
   - probability gauge
   - radar chart
   - SHAP table
   - report download
   - recommendations
11. If user wants roadmap:
   - AI roadmap is generated
   - roadmap sections render
   - roadmap PDF download becomes available
12. If `latest_prediction` exists, `app.py` also renders the floating chat

## 6.3 Full dashboard analytics flow

1. User switches to `T&P Dashboard`
2. `dashboard.show()` renders two tabs
3. If analytics CSV is uploaded:
   - descriptive charts and tables appear
4. If batch CSV is uploaded and run:
   - each student row is scored
   - predictions are appended
   - result CSV can be downloaded

## 7. Configuration and Contracts

## 7.1 `config.yaml`

Current config:

- `paths.data`: `data/placement_cleaned.csv`
- `paths.model`: `model/best_placement_model.joblib`
- `paths.features`: `model/feature_columns.json`
- `paths.comparison`: `model/model_comparison.json`
- `model.target_column`: `Placement_Status`
- `model.drop_columns`: `["Student_ID", "Name"]`
- `model.test_size`: `0.2`
- `model.random_state`: `42`
- `model.smote_random_state`: `42`
- `app.title`: `Engineering College Placement Prediction System`
- `app.placement_threshold`: `0.5`

Important observation:

- `config.yaml` is used by preprocessing
- but not every runtime file reads all config values
- the app threshold in config is currently not consumed by the UI logic

## 7.2 Input contract for single-student prediction

`predict_student(...)` expects a dict with these keys:

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

## 7.3 Input contract for batch prediction CSV

Required columns:

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

## 8. Cross-Cutting Implementation Patterns

## 8.1 Session state usage

The student page uses `st.session_state` for:

- `selected_skills`
- `student_skills`
- `latest_prediction`
- `roadmap_result`

This is essential because Streamlit reruns the script after interactions.

## 8.2 Artifact-based coupling

Training and inference are coupled by artifacts, not by shared sklearn pipeline objects.

Artifacts used as contracts:

- model object
- feature column order
- model comparison metrics

This is a valid lightweight design, but it requires discipline:

- training encoding logic and inference encoding logic must stay compatible

## 8.3 Logging

Modules in `src/` and `train.py` use Python logging.

The Streamlit views rely more on UI errors and warnings than structured logs.

## 8.4 Error handling

- Roadmap generation retries once then falls back
- SHAP is wrapped in `try/except`
- PDF generation is wrapped in `try/except`
- Dashboard wraps its whole view in `try/except`

The app generally prefers graceful degradation over hard failure.

## 9. Important Design Decisions

## 9.1 Why `drop_first=True` was used

One-hot encoding with `drop_first=True` avoids redundant dummy columns and helps reduce perfect multicollinearity for linear models like Logistic Regression.

Example:

- `Branch_CE` is not saved as a column
- all zero branch dummy columns imply the baseline branch category

## 9.2 Why SMOTE is applied only on training data

This is the correct design:

- synthetic balancing should not contaminate the test set
- otherwise evaluation metrics would be misleading

## 9.3 Why model artifacts are loaded at module import time

Pros:

- avoids reloading the model for every user click
- keeps prediction fast after startup

Cons:

- import has side effects
- it can make testing or hot-swapping artifacts slightly less explicit

## 9.4 Why SHAP is computed in the view instead of `src/`

This is a convenience design, not a strict architecture design.

The SHAP computation is tightly coupled to presentation needs:

- top features
- human-readable risk factor
- inline display

It works, but it mixes compute-heavy logic into the UI layer.

## 10. Current Limitations and Maintenance Risks

These are the most important technical realities in the current codebase.

### 10.1 Azure API key is exposed in the browser-side chatbot

The floating chat in `views/student.py` reads the Azure key from `.env` and injects it into JavaScript for direct `fetch(...)` calls from the user's browser.

Architecturally, that means:

- the client can access the key
- usage is not mediated by the Python backend
- this is not appropriate for a secure production deployment

The roadmap generation path is safer because it uses server-side Python through `src/advisor.py`.

### 10.2 SHAP is expensive and recomputes training preprocess on each prediction

For every displayed prediction, the code:

- calls `load_and_preprocess()`
- rebuilds the training background set
- instantiates `KernelExplainer`
- computes SHAP values

This is computationally expensive for a Streamlit interaction.

The dataset is small enough that it may still feel acceptable, but the design does not scale well.

### 10.3 Preprocessing is duplicated rather than packaged

Training and inference both do manual `pd.get_dummies(..., drop_first=True)` and schema alignment.

That is acceptable in a small project, but a stronger design would persist a preprocessing pipeline together with the model.

### 10.4 Config is only partially centralized

Some behavior comes from `config.yaml`, but some values are still hard-coded elsewhere:

- model definitions live in `train.py`
- branch and gender options live in `views/student.py`
- threshold logic is effectively hard-coded in the UI

### 10.5 Label mismatch for Civil branch

The real dataset branch category is `CE`, and because of `drop_first=True`, that branch becomes the baseline with no explicit dummy column.

However, both:

- `views/student.py`
- `src/report_gen.py`

contain labels for `Branch_CIVIL`, which is not present in the saved feature schema.

This does not break prediction because the feature is never expected, but it is a maintenance inconsistency.

### 10.6 Some imports and dependencies are unused in the main flow

Examples:

- `matplotlib.pyplot` in `views/student.py`
- `io` imported in both views
- some packages in `requirements.txt` are not used by the current runtime path

This does not break functionality, but it indicates some drift.

### 10.7 `config.yaml` placement threshold is not actually driving prediction UI

`config.yaml` contains:

- `app.placement_threshold: 0.5`

But the prediction page uses direct comparisons against `0.5` in charts and result semantics rather than reading that value from config.

## 11. How the Prediction Itself Works

The actual classification logic is simple:

1. Convert a student's form input into a tabular row
2. Encode categorical fields into dummy variables
3. Align columns to the saved training feature order
4. Feed the vector to the trained Logistic Regression model
5. Read:
   - `predict_proba(...)[0][1]` as placement probability
   - `predict(...)[0]` as final 0 or 1 class

Because Logistic Regression is currently the active model, each feature contributes linearly to the log-odds of placement after encoding.

The app does not display raw coefficients, but SHAP is used to approximate per-feature contribution for an individual prediction.

## 12. How the Explainability Layer Works

The SHAP subsystem is local and per-student.

Inputs:

- the current trained model
- the current student's encoded row
- a sampled background set from training data

Mechanism:

- `shap.KernelExplainer`

Output:

- per-feature SHAP values
- sorted by absolute magnitude

Interpretation:

- positive SHAP value pushes the prediction toward `Placed`
- negative SHAP value pushes it toward `Not Placed`

The UI then uses the most negative feature as the "Key Risk Factor".

## 13. How the Advisory Layer Works

There are two different advisory subsystems:

### 13.1 Rule-based recommendations

Location:

- `views/student.py`
- `src/report_gen.py`

Behavior:

- purely threshold-based
- deterministic
- fast

Examples:

- backlogs > 0 => warning
- projects < 2 => warning
- aptitude < 6 => info

### 13.2 AI-based roadmap

Location:

- `src/roadmap_gen.py`

Behavior:

- uses student profile
- uses selected skill list
- uses top SHAP factors
- injects field-specific resources
- asks Azure OpenAI for a strict JSON roadmap

This design is stronger than a generic chatbot because it combines:

- ML output
- explainability
- domain heuristics
- LLM generation

## 14. How the Chatbot Differs from the Roadmap Generator

This distinction matters:

### Chatbot

- implemented inside `views/student.py`
- browser-side JavaScript
- direct browser call to Azure OpenAI
- conversational and open-ended
- not JSON constrained

### Roadmap generator

- implemented in `src/roadmap_gen.py`
- server-side Python
- goes through `src/advisor.py`
- structured JSON output
- post-processed and rendered by Python

So the repository has two separate LLM integration patterns.

## 15. Extension Points

If someone wants to extend this project, these are the natural places.

### Add a new model

- edit `train.py -> get_models()`
- retrain
- regenerate artifacts

### Change preprocessing

- edit `src/preprocess.py`
- ensure `src/predict.py` remains compatible
- retrain and regenerate `feature_columns.json`

### Add a new student input field

1. Add the column to the training dataset
2. Update preprocessing
3. Retrain
4. Update student form in `views/student.py`
5. Update batch CSV contract in `views/dashboard.py`
6. Update report sections if needed

### Change AI roadmap behavior

- edit `FIELD_RESOURCES`
- edit `detect_field(...)`
- edit prompt template in `generate_roadmap(...)`

### Improve PDFs

- edit `src/report_gen.py`
- or `src/roadmap_pdf.py`

## 16. Practical Mental Model for Maintainers

If you need to understand this codebase quickly, think of it in this order:

1. `train.py`
   - creates the deployable artifacts

2. `src/preprocess.py`
   - defines the training feature engineering contract

3. `src/predict.py`
   - reconstructs that contract at inference time

4. `views/student.py`
   - orchestrates the user-facing prediction workflow

5. `views/dashboard.py`
   - handles analytics and batch inference

6. `src/roadmap_gen.py`, `src/report_gen.py`, `src/roadmap_pdf.py`
   - add explanation, advisory, and document-generation layers on top of the prediction

That is the real architecture.

## 17. Short Conclusion

This project is a direct and functional ML application with a clear artifact-driven flow:

- training produces model artifacts
- the Streamlit app consumes those artifacts
- the student page wraps prediction with explainability, recommendations, AI guidance, and PDF exports
- the dashboard adds descriptive analytics and batch inference

Its strengths are:

- simple structure
- readable flow
- end-to-end completeness
- practical feature set

Its main weaknesses are:

- browser-exposed chatbot credentials
- repeated preprocessing logic
- SHAP cost inside the UI
- some configuration and labeling inconsistencies

Even with those issues, the codebase is easy to reason about because most flows are explicit and there are very few hidden abstractions.
