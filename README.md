# README

## Overview
This repository contains workflows for three main tasks:  
1. **LifeTime Ratio Analysis:** Analyzing the frequency trends of scientific terms and generating LifeTime Ratios.  
2. **FICE Calculation for 2015 Papers:** Using FICE (First Innovation Contribution Extent) to correlate scientific entity novelty with citation counts.  
3. **Scientific Entity Extraction Comparison:** Comparing model-extracted entities with human annotations.

---

## **1. LifeTime Ratio Analysis**

### **Description**
This folder contains scripts to analyze the frequency of scientific terms over time and generate LifeTime Ratios. The merged output is available for direct use at: 'FICE_Area_Preparation/parallel/area_csv_file/merged_area_file.csv'


### **Steps**
1. **`1_modify_bibtex_to_csv.py`:**  
   Extracts data from BibTeX files, including Year, Title, and URL (for citation count retrieval).  

2. **`2_gpt_extract.py`:**  
   Uses GPT-based models to extract scientific entities from paper titles. Requires an OpenAI API key to run.

3. **`3_disambiguous.py`:**  
   Clusters similar scientific terms by analyzing their semantic similarity and assigns a representative term for each cluster. Updates the CSV with disambiguated terms.

4. **`4_separate_for_parallel.py`:**  
   Splits the data for parallel processing. Adjust `device_total_number` based on your system's thread capacity.

5. **`5_LifeTime_Ratio_Generation.py`:**  
   Analyzes the frequency of scientific terms over time, fits Gaussian models, detects peaks in word trends, and visualizes results. Supports parallel processing; adjust `device_total_number` accordingly.

6. **`6_combined_results.py`:**  
   Merges processed data, filters invalid entries (e.g., negative ratios), caps ratios at 1, and outputs a cleaned CSV file.

---

## **2. FICE Calculation for 2015 Papers**

### **Description**
This folder contains scripts to calculate FICE scores for papers published in 2015 and analyze the correlation between novelty (FICE) and citation counts over five years (2015–2020).

### **Steps**
1. **`1_filter_year_2015.py`:**  
   Extracts papers published in 2015 from the dataset.

2. **`2_get_citations.py`:**  
   Retrieves citation counts for each paper (2015–2020) using the paper's URL. Requires an ACL API key to run.

3. **`3_retrieval_Scientific_entity.py`:**  
   Extracts scientific entities for each paper after disambiguation.

4. **`4_calculate_FICE.py`:**  
   Calculates FICE scores for papers published in 2015.

5. **`5_draw_diagram.py`:**  
   - Groups papers by citation counts.
   - Calculates average FICE scores and citations (C₅) per bin.  
   - Applies log transformation to C₅, computes Pearson correlation, and visualizes the relationship with a plot including error bars and correlation details.  
   - Generates diagrams similar to Figure 5 in the paper.

6. **`draw_Cognitive_vs_Year.py`:**  
   - Analyzes the cognitive extent (unique scientific entities) across different years.  
   - Bins data into sizes of 125, 250, and 500, and calculates the number of unique entities per bin.  
   - Fits polynomial regression curves and visualizes trends for disambiguated and non-disambiguated entities.  
   - Generates Figure 4 in the paper.

7. **`draw_Count_vs_Year.py`:**  
   - Tracks yearly trends in scientific entities and paper counts.  
   - Highlights total and first-time unique entities with polynomial-fitted trends.  
   - Generates Figure 2 in the paper.

---

## **3. Scientific Entity Extraction Comparison**

### **Description**
This folder evaluates the quality of scientific entity extraction by comparing outputs from models (e.g., SciBERT, Spacy, GPT-4) with human annotations.

### **Steps**
1. **Human Annotation Comparison:**  
   Includes manually extracted scientific entities for evaluation.

2. **Model Outputs:**  
   Contains extractions from:
   - SciBERT  
   - Spacy  
   - GPT-4  

3. **Evaluation Metrics:**  
   - **Precision**  
   - **Recall**  
   - **F1-Score**  
   These metrics assess the alignment of model outputs with human annotations.

---

## **Notes**
- Ensure all required API keys (e.g., OpenAI, ACL) are configured correctly before running scripts.  
- Adjust `device_total_number` in parallel processing scripts based on your system's capabilities.  
- Precomputed data and merged outputs are available in respective folders for direct use if you wish to skip intermediate steps.

