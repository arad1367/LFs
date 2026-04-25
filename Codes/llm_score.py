import os
import glob
import pandas as pd
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
file_paths = glob.glob("WhitePapers_TXT/*.txt")

system_prompt = (
    "You are an expert quantitative financial analyst evaluating a cryptocurrency whitepaper "
    "for an academic study. Evaluate the text based on 18 latent factors. For each factor, "
    "output an integer score strictly from 0 to 3 (0=Absent, 1=Weak/Generic, 2=Moderate, "
    "3=Strong/Well-evidenced). Provide a direct 'LLM_Rating' from 1.0 to 5.0 representing "
    "overall assessment. Provide a brief qualitative 'LLM_Reason' for the rating and "
    "'Brief_Evidence' summarizing the presence of the 18 factors."
)

for file_path in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()[:80000] 
            
        response = client.beta.chat.completions.parse(
            model="gpt-5.4-mini-2026-03-17",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": document_text}
            ],
            response_format=WhitepaperEvaluation,
            temperature=0.0
        )
        
        evaluation_dict = response.choices[0].message.parsed.model_dump()
        evaluation_dict['Paper_Number'] = os.path.basename(file_path)
        results.append(evaluation_dict)
        
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")

df = pd.DataFrame(results)
columns_order = ['Paper_Number', 'LLM_Rating', 'LLM_Reason', 'Brief_Evidence'] + \
                [col for col in df.columns if col.startswith('LF')]
df = df[columns_order]
df.to_csv("llm_extracted_factors.csv", index=False)