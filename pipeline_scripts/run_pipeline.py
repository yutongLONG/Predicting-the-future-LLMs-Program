#!/usr/bin/env python3
"""
Election Prediction Pipeline - SLURM Version
Running two large models for election prediction analysis
"""

import re
import os
import json
import sys
import torch
import pandas as pd
from typing import List
from datetime import datetime

# Add the necessary imports
from surveygen.survey_manager import (
    SurveyOptionGenerator,
    conduct_survey_question_by_question,
)
from surveygen.llm_interview import LLMInterview
from surveygen.inference.survey_inference import default_model_init
from surveygen.inference.response_generation import JSONResponseGenerationMethod
from surveygen.utilities import constants, placeholder
from surveygen.utilities.utils import create_one_dataframe
from surveygen.parser import json_parse_all

# Set the working directory (adjust as needed)
WORK_DIR = os.getcwd()
print(f"Working directory: {WORK_DIR}")

# Model configuration 
print("INITIALIZING MODELS")


import torch
from vllm import LLM

# Check GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"✓ Found {num_gpus} GPU(s)")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available!")
    sys.exit(1)

# Meta Llama 3.3 70B Instruct
print("\nLoading Llama 3.3 70B...")
model_llama = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    max_model_len=5000,
    seed=42,
    tensor_parallel_size=num_gpus
)

# Mixtral 8x7B Instruct
print("Loading Mixtral 8x7B...")
model_mixtral = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_model_len=5000,
    seed=42,
    tensor_parallel_size=num_gpus
)

# Store the two models in a dictionary for later use
models = {
    "llama-3.3-70b": model_llama,
    "mixtral-8x7b": model_mixtral
}

print(f"✓ Models loaded: {list(models.keys())}")

# Configure the survey and tasks 
print("SETTING UP INTERVIEWS")


survey_path = "president.csv"
if not os.path.exists(survey_path):
    print(f"ERROR: {survey_path} not found!")
    sys.exit(1)

system_prompt = "You are a professional U.S. election market analyst specializing in political betting data and prediction modeling."

historical_months = [
    "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09", "2024-10"
]

final_prediction_month = "2024-11"
all_months = historical_months + [final_prediction_month]
election_to_predict = "betting odds for 2024 US Presidential Election"

# Generate the task descriptions
formatted_tasks_historical = [
    f"Analyze the betting odds for the {election_to_predict} as of {month}. "
    f"Identify the top 5 candidates based on betting market probabilities in {month}."
    for month in historical_months
]

formatted_task_final = f"Provide your FINAL PREDICTION for the {election_to_predict} as of {final_prediction_month}. This is your definitive forecast before Election Day."

# Create the interviews
interviews_historical: List[LLMInterview] = []

for task, month in zip(formatted_tasks_historical, historical_months):
    interviews_historical.append(
        LLMInterview(
            interview_source=survey_path,
            interview_name=f"historical_{month}",
            system_prompt=system_prompt,
            prompt=task,
            seed=42,
        )
    )

interview_final = LLMInterview(
    interview_source=survey_path,
    interview_name=f"FINAL_{final_prediction_month}",
    system_prompt=system_prompt,
    prompt=formatted_task_final,
    seed=42,
)

interviews_all = interviews_historical + [interview_final]

print(f"✓ Created {len(interviews_historical)} historical monthly interviews (Jan-Oct)")
print(f"✓ Created 1 final prediction interview (Nov)")

# JSON response configuration
json_fields = {
    "month": "The month being analyzed (format: YYYY-MM)",
    "predictions": "Array of exactly 5 objects, each containing: candidate (name), party (affiliation), and betting_odds_percentage (number). The betting_odds_percentage values across all 5 objects MUST sum to approximately 100%"
}

answer_production_method = JSONResponseGenerationMethod(
    json_fields=json_fields,
    output_template="""Respond only in JSON format with this structure:
{
  "month": "YYYY-MM",
  "predictions": [
    {"candidate": "Name1", "party": "Party1", "betting_odds_percentage": XX.X},
    {"candidate": "Name2", "party": "Party2", "betting_odds_percentage": XX.X},
    {"candidate": "Name3", "party": "Party3", "betting_odds_percentage": XX.X},
    {"candidate": "Name4", "party": "Party4", "betting_odds_percentage": XX.X},
    {"candidate": "Name5", "party": "Party5", "betting_odds_percentage": XX.X}
  ]
}

CRITICAL: The sum of all betting_odds_percentage values must equal approximately 100%.""",
    automatic_output_instructions=True,
    output_index_only=False
)

# Prepare all interviews
for interview in interviews_all:
    interview.prepare_interview(
        question_stem="Provide predictions as an array of objects, where each object contains candidate name, party, and betting odds percentage. Ensure the percentages sum to 100%.",
        answer_options=None,
        randomized_item_order=False,
    )

print(f"✓ All interviews prepared successfully")

# Run model inference 
print("RUNNING MODEL INFERENCE")


all_model_results = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"RUNNING MODEL: {model_name}")
    print('='*60)
    
    # Phase 1: Historical months
    print(f"\nPHASE 1: Historical Monthly Predictions (Jan-Oct) - {model_name}")
    results_historical = conduct_survey_question_by_question(
        model,
        interviews=interviews_historical,
        print_conversation=True,
        seed=42,
        temperature=0.7,
        max_tokens=2000,
    )
    print(f"✓ Completed {len(results_historical)} historical predictions")
    
    # Phase 2: Final prediction
    print(f"\nPHASE 2: Final Prediction (November 2024) - {model_name}")
    results_final = conduct_survey_question_by_question(
        model,
        interviews=[interview_final],
        print_conversation=True,
        seed=42,
        temperature=0.7,
        max_tokens=2000,
    )
    print(f"✓ Completed final prediction")
    
    # Merge the results
    results_all = results_historical + results_final
    all_model_results[model_name] = results_all

# Parse the results 
print("PARSING RESULTS")


all_parsed_results = {}

for model_name, results_all in all_model_results.items():
    print(f"\nPARSING RESULTS: {model_name}")
    
    parsed_response_all = json_parse_all(results_all)
    
    parsed_historical = {interview: parsed_response_all[interview] for interview in interviews_historical}
    parsed_final = parsed_response_all[interview_final]
    
    all_parsed_results[model_name] = {
        "all": parsed_response_all,
        "historical": parsed_historical,
        "final": parsed_final
    }
    
    print(f"✓ {model_name} parsed successfully")

print("\n✓ All models parsed successfully")

# Create DataFrame 
print("CREATING DATAFRAMES")

all_model_dataframes = {}

for model_name, parsed_data in all_parsed_results.items():
    print(f"\nCREATING DATAFRAMES: {model_name}")
    
    df_historical = create_one_dataframe(parsed_data["historical"])
    df_historical['data_type'] = 'historical'
    df_historical['model'] = model_name
    
    parsed_final = parsed_data["final"]
    if isinstance(parsed_final, pd.DataFrame):
        df_final = parsed_final.copy()
    elif isinstance(parsed_final, dict):
        df_final = pd.DataFrame(parsed_final)
    else:
        df_final = pd.DataFrame([parsed_final])
    
    df_final['data_type'] = 'final_prediction'
    df_final['model'] = model_name
    
    df_complete = pd.concat([df_historical, df_final], ignore_index=True)
    
    all_model_dataframes[model_name] = {
        "historical": df_historical,
        "final": df_final,
        "complete": df_complete
    }
    
    print(f"✓ {model_name} - Historical: {len(df_historical)} rows, Final: {len(df_final)} rows")

# Merge all models' data
df_all_models = pd.concat([
    data["complete"] for data in all_model_dataframes.values()
], ignore_index=True)

print(f"\n✓ Combined DataFrame: {len(df_all_models)} total rows")

# Data validation 
print("VALIDATION")


for model_name, dfs in all_model_dataframes.items():
    print(f"\nVALIDATION: {model_name}")
    
    df_historical = dfs["historical"]
    df_final = dfs["final"]
    
    print(f"HISTORICAL MONTHS (Jan-Oct) - {model_name}:")
    for month in historical_months:
        month_data = df_historical[df_historical['month'] == month]
        total = month_data['betting_odds_percentage'].sum()
        status = "✓" if 95 <= total <= 105 else "⚠"
        print(f"  {status} {month}: {total:.1f}%")
    
    print(f"FINAL PREDICTION (Nov) - {model_name}:")
    final_total = df_final['betting_odds_percentage'].sum()
    status = "✓" if 95 <= final_total <= 105 else "⚠"
    print(f"  {status} {final_prediction_month}: {final_total:.1f}%")

# Save results 
print("SAVING RESULTS")


output_dir = "prediction_results"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for model_name, dfs in all_model_dataframes.items():
    print(f"\nSAVING: {model_name}")
    
    model_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    dfs["historical"].to_csv(f"{model_dir}/historical_{timestamp}.csv", index=False)
    dfs["final"].to_csv(f"{model_dir}/final_{timestamp}.csv", index=False)
    dfs["complete"].to_csv(f"{model_dir}/complete_{timestamp}.csv", index=False)
    
    with open(f"{model_dir}/parsed_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(
            all_parsed_results[model_name]["all"], 
            f, 
            indent=2, 
            ensure_ascii=False, 
            default=str
        )
    
    print(f"✓ {model_name} - All files saved")

df_all_models.to_csv(f"{output_dir}/all_models_combined_{timestamp}.csv", index=False)

comparison_summary = {
    "timestamp": timestamp,
    "models": list(all_model_dataframes.keys()),
    "predictions": {}
}

for model_name, dfs in all_model_dataframes.items():
    df_final = dfs["final"]
    winner = df_final.loc[df_final['betting_odds_percentage'].idxmax()]
    comparison_summary["predictions"][model_name] = {
        "winner": winner['candidate'],
        "probability": float(winner['betting_odds_percentage']),
        "top_5": df_final.nlargest(5, 'betting_odds_percentage')[
            ['candidate', 'party', 'betting_odds_percentage']
        ].to_dict('records')
    }

with open(f"{output_dir}/model_comparison_{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump(comparison_summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✓ All data saved to {output_dir}/")
print("PIPELINE COMPLETED SUCCESSFULLY")
