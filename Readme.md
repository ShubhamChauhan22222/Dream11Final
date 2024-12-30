# Dream 11 Next-Gen Team Builder With Predictive AI

Welcome to the Fantasy Cricket Team Recommendation Project repository! This project is designed to predict the optimal fantasy cricket team for upcoming matches, leveraging advanced data processing, modeling, and a user-friendly interface.

---

## 🚀 Main Application Workflow
The primary script, `main_app.py`, orchestrates the following workflow:
2. **UI Launch**: Opens the main user interface, `Final_UI.py`.
3. **UI Options**:
   - **ModelUI**: Opens a Streamlit-based web app for model interactions.
   - **ProductUI**: Downloads the `app.apk` for the mobile application.

---

## 📂 Repository Structure

### Root Files
- **`README.md`**: Overview and usage instructions for the project.
- **`main_app.py`**: The main entry point for running the project.

### Data Folder (`data`)
- **`raw`**: Contains original data as downloaded.
  - `cricksheet_data`: Raw data sourced from Cricksheet.
  - `additional_data`: Raw data from other sources (if applicable).
- **`interim`**: Intermediate data files generated during processing.
- **`processed`**: Finalized datasets ready for modeling.

### Data Processing Folder (`data_processing`)
- **`data_download.py`**: Script to download and organize all raw data.
- **`feature_engineering.py`**: Performs data manipulation and feature engineering.

### Documentation Folder (`docs`)
- **`video_demo`**: Contains a walkthrough video covering setup, UI, and functionality.

### Model Folder (`model`)
- **`train_model.py`**: Script to train models using the processed data.
- **`predict_model.py`**: Script to generate predictions using trained models.

### Model Artifacts Folder (`model_artifacts`)
- Stores all trained models, including pre-trained models for `ProductUI` and models integrated with `ModelUI`.

### Out-of-Sample Data Folder (`out_of_sample_data`)
- Contains dummy evaluation data for matches.
- Post-submission (Dec 4–14), testing data will be added in the same format as the provided sample data.
- Integrated with `ModelUI` to append new data automatically from Cricksheet.

### Miscellaneous Folder (`rest`)
- Contains files for any additional requirements not covered in other directories.

### UI Folder (`UI`)
- **`Final_UI.py`**: Main user interface for selecting options.
- **`Model_UI.py`**: Streamlit app for model-related interactions.
- **`Product_UI.py`**: Facilitates the download of the `app.apk` for mobile use.

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.8 or higher.
2. Required Python packages (install using `pip install -r requirements.txt`).
3. Streamlit installed for running `ModelUI`.

### Running the Application
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Run the main script:
   ```bash
   python main_app.py
   ```
3. Follow the on-screen instructions to:
   - Update data and launch the main UI.
   - Choose between:
     - **ModelUI**: Opens a Streamlit-based web app for model interactions.
     - **ProductUI**: Automatically downloads the `app.apk`.

---

## 🛠 Future Improvements
- Enhance data processing for real-time updates.
- Expand UI functionalities for better user interaction.
- Add support for additional data sources.

---

## 📧 Support
For queries, please reach out to the project maintainers.



