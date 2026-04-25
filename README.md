[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/)
[![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/arad1367)

# LFs: Latent Factor Extraction for Blockchain Whitepapers

This repository contains the dataset, extraction scripts, and analysis code for the academic research paper evaluating the quality of cryptocurrency whitepapers using Large Language Models (LLMs).

## Overview

Evaluating early-stage blockchain ventures is frequently hindered by extreme information asymmetry and highly technical jargon. This project develops an LLM-assisted evaluation framework that transforms unstructured whitepaper text into a structured set of 18 theoretically derived Latent Factors (LFs).

The repository provides the code to:

1. Extract and clean unstructured text from PDF whitepapers.
2. Prompt an LLM (gpt-5.4-mini-2026-03-17) to structurally evaluate the text using both Zero-Shot and Few-Shot paradigms.
3. Run econometric (OLS) and non-linear machine learning (Random Forest) models to test whether these AI-extracted variables explain and predict human investor ratings.

## Data Source

The primary documents and baseline human ratings utilized in this study are derived from the **Token Offerings Research Database (TORD)**. The sample consists of 312 English-language cryptocurrency whitepapers (ICOs, IEOs, and STOs).

_Note: Due to copyright and size limitations, the raw `.pdf` and `.txt` files of the whitepapers are not included in this repository. The `data/` folder contains the processed, extracted latent factor scores necessary to reproduce the regression analyses._

## Methodology

The taxonomy consists of 18 Latent Factors organized into five higher-order categories:

1. Project Narrative & Disclosure Quality
2. Business Logic & Market Execution
3. Technical Feasibility & Architecture
4. Tokenomics & Financial Viability
5. Governance, Compliance, & Ecosystem

The LLM is prompted to score each factor on an ordinal scale (0 = absent to 3 = strong/well-evidenced) and provide qualitative reasoning for its outputs to ensure interpretability and traceability.

## Getting Started

### Prerequisites

Clone the repository and install the required packages:

```bash
git clone https://github.com/arad1367/LFs.git
cd LFs
pip install -r requirements.txt
```

_(Ensure you have `pycryptodome` installed to handle AES-encrypted PDFs, as well as `pandas`, `scikit-learn`, `statsmodels`, and `openai`)_.


## Citation

If you utilize this code, taxonomy, or dataset in your research, please cite the forthcoming paper. (Citation details will be updated upon publication).

For inquiries or issues, please open an issue on this repository or contact the author: pejman.ebeahimi@uni.li or pejman.ebrahimi77@gmail.com
