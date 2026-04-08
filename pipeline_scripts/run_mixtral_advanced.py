#!/usr/bin/env python3
"""
Election Prediction Pipeline - Verbalized Distribution Version
Have the model automatically select the top 5 from the candidate pool and predict their respective percentages.
"""

import os
import json
import pandas as pd
from typing import List
from datetime import datetime
import calendar

# QSTN imports
from qstn.prompt_builder import LLMPrompt
from qstn import survey_manager
from qstn import parser
from qstn.utilities import placeholder
from qstn.utilities import create_one_dataframe
from qstn.inference.response_generation import JSONVerbalizedDistribution

if __name__ == "__main__":
    
    # Set the working directory
    WORK_DIR = os.getcwd()
    print(f"Working directory: {WORK_DIR}")

    # Model Initialization
    print("\nINITIALIZING MODEL")
    from vllm import LLM
    num_gpus = 2
    
    model = LLM(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_model_len=2048,
        seed=42,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True
    )
    print(f"✓ Model loaded: Mixtral 8x7B Instruct")

    # Configuration Questionnaire
    print("\nSETTING UP QUESTIONNAIRES")
    
    system_prompt = "You are an expert political analyst specializing in U.S. election betting markets."
    
    # Define the candidate pool (16 candidates)
    candidate_list = [
        "Biden", "Trump", "Harris", "Haley", "Newsom", "Kennedy", "DeSantis",
        "Ramaswamy", "Christie", "Warren", "Sanders", "AOC", "Clinton", 
        "Stein", "West", "Oliver"
    ]

    # Define the questionnaire content
    questionnaire = pd.DataFrame([
        {"questionnaire_item_id": 1, "question_content": "Top 5 Candidates and Their Betting Odds"}
    ])

    # Define time points (historical months + final prediction)
    historical_months = []
    for month_num in range(1, 11):  # January to October
        year = 2024
        last_day = calendar.monthrange(year, month_num)[1]
        historical_months.append(f"{year}-{month_num:02d}-{last_day:02d}")
    
    final_prediction_month = "2024-11-05"
    all_months = historical_months + [final_prediction_month]
    
    # Create a list of prediction tasks
    elections_to_predict = [
        f"2024 US Presidential Election as of {month}" 
        for month in all_months
    ]

    # Define the Response Generation Method (key)
    response_generation_method = JSONVerbalizedDistribution(
        output_template=(
            "Respond only in JSON format with this exact structure:\n"
            "{\n"
            '  "month": "YYYY-MM-DD",\n'
            '  "predictions": [\n'
            '    {"candidate": "Name1", "party": "Party1", "betting_odds_percentage": XX.X},\n'
            '    {"candidate": "Name2", "party": "Party2", "betting_odds_percentage": XX.X},\n'
            '    {"candidate": "Name3", "party": "Party3", "betting_odds_percentage": XX.X},\n'
            '    {"candidate": "Name4", "party": "Party4", "betting_odds_percentage": XX.X},\n'
            '    {"candidate": "Name5", "party": "Party5", "betting_odds_percentage": XX.X}\n'
            '  ]\n'
            "}\n"
            "CRITICAL RULES:\n"
            "1. Output EXACTLY 5 candidates (no more, no less)\n"
            "2. All percentages must sum to 100%\n"
            "3. Select candidates from the provided list based on betting market data"
        ),
        output_index_only=False,
    )
    
    # Candidate Pool Construction Prompt Text
    candidate_pool_text = (
        f"Consider the following potential candidates: "
        f"{', '.join(candidate_list)}."
    )

    # Build prompts (including candidate pool and output instructions)
    formatted_tasks = [
        (
            f"Analyze the betting odds for the {election}. "
            f"{candidate_pool_text} "
            f"Based on betting market probabilities, identify the top 5 candidates with the highest odds. "
            f"{response_generation_method.output_template}\n"
            f"{placeholder.PROMPT_QUESTIONS}"
        )
        for election, month in zip(elections_to_predict, all_months)
    ]

    # Create interviews
    interviews: List[LLMPrompt] = []
    for task, election, month in zip(formatted_tasks, elections_to_predict, all_months):
        interviews.append(
            LLMPrompt(
                questionnaire_source=questionnaire,
                questionnaire_name=election,
                system_prompt=system_prompt,
                prompt=task,
                seed=42,
            )
        )
    
    print(f"✓ Created {len(interviews)} questionnaires (Jan-Nov)")
    
    # View the first prompt (optional, for debugging)
    print("\n" + "="*60)
    print("EXAMPLE PROMPT (January):")
    print("="*60)
    system, prompt = interviews[0].get_prompt_for_questionnaire_type()
    print(f"System: {system}")
    print(f"Prompt: {prompt[:500]}...")
    
    # Operational Reasoning
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    results = survey_manager.conduct_survey_single_item(
        model,
        llm_prompts=interviews,
        client_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        print_conversation=True,
        temperature=0.5,
        max_tokens=1000,
        seed=42,
    )
    print(f"✓ Completed all predictions")
    
    # Parsing Results (using JSON parser)
    print("\nPARSING RESULTS")
    parsed_response = parser.parse_json(results)
    print(f"✓ Results parsed successfully")
    
    # Create complete DataFrame
    print("\nCREATING DATAFRAMES")
    df_complete = create_one_dataframe(parsed_response)
    print(f"✓ Created DataFrame with {len(df_complete)} rows")
    
    # Show sample results
    print("\n" + "="*60)
    print("SAMPLE RESULTS (First 3 months):")
    print("="*60)
    if not df_complete.empty:
        print(df_complete.head(15).to_markdown())  # 每个月5行，显示3个月
    
    # Data Validation
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    # Monthly checks
    for month in all_months:
        month_data = df_complete[df_complete['questionnaire_name'].str.contains(month)]
        if not month_data.empty:
            # Assuming percentages are present in certain columns, adjustments may be required based on actual parsing results.
            # Here, the actual structure of `parsed_response` needs to be examined.
            print(f"  ✓ {month}: {len(month_data)} candidates found")
    
    # Save Results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_dir = "prediction_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = f"{output_dir}/mixtral-8x7b-verbalized-dist"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save DataFrame
    df_complete.to_csv(f"{model_dir}/complete_{timestamp}.csv", index=False)
    
    # Save original parsed response (for debugging)
    with open(f"{model_dir}/parsed_response_{timestamp}.json", "w", encoding="utf-8") as f:
        # Convert to serializable format
        serializable_data = {}
        for key, value in parsed_response.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[str(key)] = value.to_dict('records')
            else:
                serializable_data[str(key)] = str(value)
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ All files saved to {model_dir}/")
    
    # Completion abstract
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nResults saved to: {model_dir}/")
    print(f"Total predictions: {len(df_complete)} rows")