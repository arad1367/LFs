import os
import glob
import pandas as pd
import json
from pydantic import BaseModel, Field
from openai import OpenAI

class WhitepaperEvaluation(BaseModel):
    LF1_Transparency: int = Field(ge=0, le=3)
    LF2_RoadmapCredibility: int = Field(ge=0, le=3)
    LF3_RiskDisclosure: int = Field(ge=0, le=3)
    LF4_Differentiation: int = Field(ge=0, le=3)
    LF5_Traction: int = Field(ge=0, le=3)
    LF6_BusinessModelClarity: int = Field(ge=0, le=3)
    LF7_GoToMarket: int = Field(ge=0, le=3)
    LF8_ExecutionCredibility: int = Field(ge=0, le=3)
    LF9_TechnicalFeasibility: int = Field(ge=0, le=3)
    LF10_SmartContractSecurity: int = Field(ge=0, le=3)
    LF11_Scalability: int = Field(ge=0, le=3)
    LF12_IncentiveCompatibility: int = Field(ge=0, le=3)
    LF13_FinancialSkewness: int = Field(ge=0, le=3)
    LF14_TokenomicsSustainability: int = Field(ge=0, le=3)
    LF15_UseTestCaseNecessity: int = Field(ge=0, le=3)
    LF16_GovernanceDesign: int = Field(ge=0, le=3)
    LF17_RegulatoryExposure: int = Field(ge=0, le=3)
    LF18_PartnershipQuality: int = Field(ge=0, le=3)
    LLM_Rating: float = Field(ge=1.0, le=5.0)
    LLM_Reason: str
    Brief_Evidence: str

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
results = []
txt_folder = "WhitePapers_TXT"
file_paths = glob.glob(f"{txt_folder}/*.txt")

# Helper function to load example texts
def load_example_text(filename):
    path = os.path.join(txt_folder, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()[:80000]
    return "Error: File not found."

print("Loading Few-Shot Examples...")
# Load the 3 specific papers chosen from the analysis
example_1_text = load_example_text("paper_132.txt") # High Rating (4.7)
example_2_text = load_example_text("paper_279.txt") # Low Rating (2.1)
example_3_text = load_example_text("paper_44.txt")  # Medium Rating (3.2)

system_prompt = (
    "You are an expert quantitative financial analyst evaluating a cryptocurrency whitepaper "
    "for an academic study. Evaluate the text based on 18 latent factors. For each factor, "
    "output an integer score strictly from 0 to 3 (0=Absent, 1=Weak, 2=Moderate, 3=Strong). "
    "Crucially, align your 'LLM_Rating' with the standard of the provided human-rated examples. "
    "Provide a brief qualitative 'LLM_Reason' and 'Brief_Evidence'."
)

# Example 1 (High: 4.7) 
example_1_output = {
    "LF1_Transparency": 1, "LF2_RoadmapCredibility": 1, "LF3_RiskDisclosure": 0, "LF4_Differentiation": 2,
    "LF5_Traction": 2, "LF6_BusinessModelClarity": 2, "LF7_GoToMarket": 1, "LF8_ExecutionCredibility": 1,
    "LF9_TechnicalFeasibility": 1, "LF10_SmartContractSecurity": 1, "LF11_Scalability": 2, "LF12_IncentiveCompatibility": 1,
    "LF13_FinancialSkewness": 1, "LF14_TokenomicsSustainability": 1, "LF15_UseTestCaseNecessity": 2, "LF16_GovernanceDesign": 0,
    "LF17_RegulatoryExposure": 1, "LF18_PartnershipQuality": 1,
    "LLM_Rating": 4.7, 
    "LLM_Reason": "The paper presents a concrete but highly promotional concept centered on a BD token tied to forex brokerage, PAMM, and copy-trading services. The business model is understandable, yet execution credibility and tokenomics sustainability are not well supported.",
    "Brief_Evidence": "Transparency is low with sparse verifiable disclosures; roadmap exists but is generic; no meaningful risk section; differentiation comes from cross-broker copy trading and forex-token integration; traction claims include 6000 accounts but are not evidenced."
}

# Example 2 (Low: 2.1) 
example_2_output = {
    "LF1_Transparency": 2, "LF2_RoadmapCredibility": 2, "LF3_RiskDisclosure": 1, "LF4_Differentiation": 2,
    "LF5_Traction": 2, "LF6_BusinessModelClarity": 2, "LF7_GoToMarket": 2, "LF8_ExecutionCredibility": 2,
    "LF9_TechnicalFeasibility": 2, "LF10_SmartContractSecurity": 1, "LF11_Scalability": 1, "LF12_IncentiveCompatibility": 2,
    "LF13_FinancialSkewness": 2, "LF14_TokenomicsSustainability": 1, "LF15_UseTestCaseNecessity": 2, "LF16_GovernanceDesign": 1,
    "LF17_RegulatoryExposure": 1, "LF18_PartnershipQuality": 2,
    "LLM_Rating": 2.1,
    "LLM_Reason": "The whitepaper presents a concrete film-backed token concept with identifiable team members, a pilot project, rough financial projections, and some distribution/partner references. However, it is promotional and incomplete.",
    "Brief_Evidence": "Transparency: preliminary/non-binding disclosures and some token parameters; Roadmap: dated milestones through 2020; Risk: minimal explicit risk discussion; Differentiation: blockchain-based film crowdfunding platform."
}

# Example 3 (Medium: 3.2) 
example_3_output = {
    "LF1_Transparency": 2, "LF2_RoadmapCredibility": 2, "LF3_RiskDisclosure": 2, "LF4_Differentiation": 3,
    "LF5_Traction": 1, "LF6_BusinessModelClarity": 3, "LF7_GoToMarket": 2, "LF8_ExecutionCredibility": 2,
    "LF9_TechnicalFeasibility": 2, "LF10_SmartContractSecurity": 1, "LF11_Scalability": 2, "LF12_IncentiveCompatibility": 3,
    "LF13_FinancialSkewness": 1, "LF14_TokenomicsSustainability": 1, "LF15_UseTestCaseNecessity": 3, "LF16_GovernanceDesign": 2,
    "LF17_RegulatoryExposure": 2, "LF18_PartnershipQuality": 0,
    "LLM_Rating": 3.2,
    "LLM_Reason": "The whitepaper presents a coherent and ambitious AR-platform thesis with a reasonably clear business model, user/advertiser segmentation, and a strong decentralization narrative. Overall it reads as conceptually strong but execution-risky.",
    "Brief_Evidence": "Transparency: moderate project description but limited verifiable data; Roadmap: staged development and architecture evolution described; Risk disclosure: some competitive/technical risks discussed; Differentiation: strong AR social network."
}

few_shot_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": example_1_text},
    {"role": "assistant", "content": json.dumps(example_1_output)},
    {"role": "user", "content": example_2_text},
    {"role": "assistant", "content": json.dumps(example_2_output)},
    {"role": "user", "content": example_3_text},
    {"role": "assistant", "content": json.dumps(example_3_output)}
]

print(f"Beginning Few-Shot evaluation on {len(file_paths)} papers...")

for i, file_path in enumerate(file_paths, 1):
    file_name = os.path.basename(file_path)
    
    # Skip the papers we used as examples so we don't evaluate them twice
    if file_name in ["paper_132.txt", "paper_279.txt", "paper_44.txt"]:
        continue
        
    print(f"[{i}/{len(file_paths)}] Processing {file_name}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()[:80000]
            
        current_messages = few_shot_messages.copy()
        current_messages.append({"role": "user", "content": document_text})
            
        response = client.beta.chat.completions.parse(
            model="gpt-5.4-mini-2026-03-17",
            messages=current_messages,
            response_format=WhitepaperEvaluation,
            temperature=0.0
        )
        
        evaluation_dict = response.choices[0].message.parsed.model_dump()
        evaluation_dict['Paper_Number'] = file_name
        results.append(evaluation_dict)
        print(f"  -> Success! Given Rating: {evaluation_dict['LLM_Rating']}")
        
        # Save a backup every 10 papers
        if i % 10 == 0:
            pd.DataFrame(results).to_csv("llm_extracted_factors_few_shot_backup.csv", index=False)
            print(f"  -> Backup saved at paper {i}")
            
    except Exception as e:
        print(f"  -> Failed processing {file_name}: {e}")

if results:
    print("\nFormatting final data...")
    df = pd.DataFrame(results)
    columns_order = ['Paper_Number', 'LLM_Rating', 'LLM_Reason', 'Brief_Evidence'] + \
                    [col for col in df.columns if col.startswith('LF')]
    df = df[columns_order]
    df.to_csv("llm_extracted_factors_few_shot.csv", index=False)
    print("All done! Saved to 'llm_extracted_factors_few_shot.csv'.")
else:
    print("No results generated.")