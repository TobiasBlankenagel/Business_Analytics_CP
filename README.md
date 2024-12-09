
# Business_Analytics_CP

## Description
This project is a comprehensive analytics platform designed to process and visualize football league data. It integrates a Jupyter Notebook for interactive analysis and a Python script for deploying an interactive web application.

### Key Features
- **Data Processing**: Clean and process football match results and league data.
- **Interactive Visualizations**: Provides rich visualizations for trends and performance.
- **Web Application**: A Streamlit-based web app for interactive user experiences.

## Files
- **Jupyter Notebook**: The notebook is used for data exploration and model training.
- **Python Script**: The python script powers an interactive web application, available at the website: https://businessanalytics-c5bqundnhifhyhmzobdngc.streamlit.app

## Dependencies
The following libraries are required for this project:
- base64
- datetime
- io
- matplotlib
- numpy
- pandas
- pickle
- requests
- streamlit

Install these dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application
1. Ensure all dependencies are installed.
2. Run the script using Streamlit:
   ```bash
   streamlit run app_v4_final.py
   ```
3. Or just open https://businessanalytics-c5bqundnhifhyhmzobdngc.streamlit.app

### Using the Jupyter Notebook
1. Open the notebook file "main.ipynb" in Jupyter Notebook or JupyterLab.
2. Execute cells interactively to explore data.

## Dataset Details
The project uses football match results and league data, which are processed and analyzed within the notebook and the web app.

### Key Files
- `football_results.csv`: Contains historical match results.
- `new_league_data.csv`: Includes additional league data displayed interactively on the website.

