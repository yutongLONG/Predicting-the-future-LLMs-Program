import os
import sys
import argparse
import json
import pandas as pd
import re
import types
import logging
import random
import hashlib
from difflib import SequenceMatcher

def configure_workspace():
    ws_path = os.environ.get("WORKSPACE_PATH")
    if not ws_path:
        cwd = os.getcwd()
        if "/pfs/work" in cwd:
            ws_path = cwd.split("workspace")[0] + "workspace"
    if ws_path and os.path.exists(ws_path):
        cache_root = os.path.join(ws_path, ".cache")
        os.environ["HF_HOME"] = os.path.join(cache_root, "huggingface")
        os.environ["TORCH_HOME"] = os.path.join(cache_root, "torch")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
        os.environ["OUTLINES_CACHE_DIR"] = os.path.join(cache_root, "outlines")
        os.environ["VLLM_CACHE_DIR"] = os.path.join(cache_root, "vllm")
        os.environ["TMPDIR"] = os.path.join(ws_path, "tmp")
        for k in ["HF_HOME", "TRITON_CACHE_DIR", "OUTLINES_CACHE_DIR", "VLLM_CACHE_DIR", "TMPDIR"]:
            os.makedirs(os.environ[k], exist_ok=True)

configure_workspace()

try:
    import pyairports
except ImportError:
    mock_airports_module = types.ModuleType("pyairports.airports")
    mock_airports_module.AIRPORT_LIST = []
    mock_pyairports_pkg = types.ModuleType("pyairports")
    mock_pyairports_pkg.__path__ = []
    mock_pyairports_pkg.airports = mock_airports_module
    sys.modules["pyairports"] = mock_pyairports_pkg
    sys.modules["pyairports.airports"] = mock_airports_module

from vllm import LLM
if hasattr(LLM, "chat"):
    original_chat = LLM.chat
    def patched_chat(self, messages, **kwargs):
        kwargs.pop("chat_template", None)
        kwargs.pop("chat_template_kwargs", None)
        return original_chat(self, messages, **kwargs)
    LLM.chat = patched_chat

from qstn.utilities import placeholder
from qstn.prompt_builder import LLMPrompt, generate_likert_options
from qstn.inference.response_generation import JSONVerbalizedDistribution
from qstn.survey_manager import conduct_survey_single_item

def load_validation_data(logger):
    validation_dir = "validation"

    polls_file = os.path.join(validation_dir, "national_polls_results.csv")
    if not os.path.exists(polls_file):
        logger.error(f"Validation file not found: {polls_file}")
        raise FileNotFoundError(f"Missing {polls_file}")
    df_polls = pd.read_csv(polls_file, sep=";", encoding='utf-8-sig')
    df_polls['poll_percentage'] = df_polls['poll_percentage'].apply(lambda x: float(str(x).replace(",", ".")))

    poll_targets = {}
    for month in df_polls['month'].unique():
        total = df_polls[df_polls['month'] == month]['poll_percentage'].sum()
        poll_targets[month] = total
    logger.info(f"Loaded polling validation data: {len(df_polls)} records, {len(poll_targets)} months")

    election_file = os.path.join(validation_dir, "national_election_results.csv")
    if not os.path.exists(election_file):
        logger.error(f"Validation file not found: {election_file}")
        raise FileNotFoundError(f"Missing {election_file}")
    df_election = pd.read_csv(election_file, sep=";", encoding='utf-8-sig')
    df_election['poll_percentage'] = df_election['poll_percentage'].apply(lambda x: float(str(x).replace(",", ".")))
    election_target = df_election['poll_percentage'].sum()
    logger.info(f"Loaded election validation data: {len(df_election)} records, sum={election_target:.2f}")

    betting_file = os.path.join(validation_dir, "national_betting_odds_results.csv")
    if not os.path.exists(betting_file):
        logger.error(f"Validation file not found: {betting_file}")
        raise FileNotFoundError(f"Missing {betting_file}")
    df_betting = pd.read_csv(betting_file, sep=";", encoding='utf-8-sig')
    df_betting['poll_percentage'] = df_betting['poll_percentage'].apply(lambda x: float(str(x).replace(",", ".")))

    betting_targets = {}
    betting_dates = df_betting['month'].unique()
    for date in betting_dates:
        total = df_betting[df_betting['month'] == date]['poll_percentage'].sum()
        betting_targets[date] = total
    logger.info(f"Loaded betting validation data: {len(df_betting)} records, {len(betting_targets)} dates")

    date_to_month = {
        "31.01.24": "January", "29.02.24": "February", "31.03.24": "March",
        "30.04.24": "April", "31.05.24": "May", "30.06.24": "June",
        "31.07.24": "July", "31.08.24": "August", "30.09.24": "September",
        "31.10.24": "October"
    }

    return {
        'poll_targets': poll_targets, 'election_target': election_target,
        'betting_targets': betting_targets, 'betting_dates': sorted(betting_dates),
        'date_to_month': date_to_month
    }

def load_finance_data(logger):
    finance_file = os.path.join("input", "campaign_finance_2024_cleaned.csv")
    if not os.path.exists(finance_file):
        logger.warning(f"Finance file not found: {finance_file}. Proceeding without finance data.")
        return pd.DataFrame()

    df_finance = pd.read_csv(finance_file, sep=";", encoding='utf-8-sig')
    name_mapping = {
        "Biden": "Joe Biden", "Trump": "Donald Trump", "Harris": "Kamala Harris",
        "Kennedy": "Robert F. Kennedy Jr.", "West": "Cornel West", "Haley": "Nikki Haley",
        "DeSantis": "Ron DeSantis", "Ramaswamy": "Vivek Ramaswamy", "Christie": "Chris Christie",
        "Hutchinson": "Asa Hutchinson", "Phillips": "Dean Phillips", "Williamson": "Marianne Williamson",
        "Palmer": "Jason Palmer", "Binkley": "Mark Binkley"
    }
    df_finance['candidate'] = df_finance['candidate'].replace(name_mapping)
    logger.info(f"Loaded finance data: {len(df_finance)} records, {df_finance['candidate'].nunique()} candidates")
    return df_finance

def load_timeline_data(logger):
    timeline_file = os.path.join("input", "timeline_2024_reclean.json")
    if not os.path.exists(timeline_file):
        logger.warning(f"Timeline file not found: {timeline_file}. Proceeding without timeline data.")
        return {}

    with open(timeline_file, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    timeline_by_month = {}
    for month_data in timeline_data.get("events", []):
        month = month_data["month"]
        month_name = month.split()[0]
        timeline_by_month[month_name] = month_data["date_events"]

    logger.info(f"Loaded timeline data: {len(timeline_by_month)} months with events")
    return timeline_by_month

def format_timeline_events(month, timeline_by_month, max_events=10):
    if not timeline_by_month or month not in timeline_by_month:
        return "No timeline data available for this month."

    date_events = timeline_by_month[month]
    all_events = []
    for date_entry in date_events:
        date = date_entry["date"]
        for event in date_entry["events"]:
            all_events.append(f"{date} {event}")

    if not all_events:
        return "No significant events recorded for this month."

    selected_events = all_events[:max_events]
    return "\n".join(f"- {event}" for event in selected_events)

def get_cumulative_timeline_summary(current_month, timeline_by_month, months_list):
    try:
        current_idx = months_list.index(current_month)
    except ValueError:
        return ""

    if current_idx == 0:
        return ""

    previous_summary = []
    for prev_month in months_list[:current_idx]:
        if prev_month not in timeline_by_month:
            continue
        date_events = timeline_by_month[prev_month]
        month_events = []
        for date_entry in date_events[:2]:
            for event in date_entry["events"][:2]:
                month_events.append(event)
                if len(month_events) >= 3:
                    break
            if len(month_events) >= 3:
                break
        if month_events:
            previous_summary.append(f"{prev_month}: {' '.join(month_events[:2])}")

    return "\n".join(previous_summary) if previous_summary else ""

def format_money(amount):
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    else:
        return f"${amount / 1000:.0f}K"

def get_candidate_finance_text(candidate_name, month, df_finance, candidates_seen_previously):
    if df_finance.empty:
        return f"{candidate_name}: No fundraising data available."

    month_data = df_finance[df_finance['month'] == month]
    if month_data.empty:
        return f"{candidate_name}: No fundraising data available for this month."

    cand_data = month_data[month_data['candidate'] == candidate_name]
    if cand_data.empty:
        if candidate_name in candidates_seen_previously:
            return f"{candidate_name}: Campaign suspended - no longer fundraising."
        else:
            return f"{candidate_name}: No fundraising data available."

    total = cand_data['total_raised'].values[0]
    quarter = cand_data['total_raised_quarter'].values[0]

    month_data_sorted = month_data.sort_values('total_raised', ascending=False).reset_index(drop=True)
    rank = month_data_sorted[month_data_sorted['candidate'] == candidate_name].index[0] + 1
    total_candidates = len(month_data_sorted)

    if rank == 1:
        rank_text = "the highest among all candidates"
    elif rank == 2:
        rank_text = "the 2nd highest among all candidates"
    elif rank == 3:
        rank_text = "the 3rd highest among all candidates"
    else:
        rank_text = f"ranked {rank} out of {total_candidates} candidates"

    return f"{candidate_name} has raised {format_money(total)} in total fundraising, {rank_text}, and raised {format_money(quarter)} in the most recent quarter."

def setup_logging(model_name, unique_id):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/v51_fixed_prompts_{model_name}_{unique_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("main")

def clean_key(key):
    s = str(key).strip()
    s = re.sub(r'[\s_-]+', ' ', s)
    return s.strip()

def fuzzy_match_candidate(key_text, candidate_pool, threshold=0.6):
    key_clean = re.sub(r'[^a-z0-9\s]', '', str(key_text).lower().strip())
    key_clean = re.sub(r'\s+', ' ', key_clean)

    best_match = None
    best_ratio = 0

    for candidate in candidate_pool:
        cand_clean = re.sub(r'[^a-z0-9\s]', '', candidate.lower().strip())
        cand_clean = re.sub(r'\s+', ' ', cand_clean)
        ratio = SequenceMatcher(None, key_clean, cand_clean).ratio()

        last_name_key = key_clean.split()[-1] if key_clean else ""
        last_name_cand = cand_clean.split()[-1] if cand_clean else ""
        if last_name_key and last_name_key == last_name_cand:
            ratio = max(ratio, 0.8)

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = candidate

    return best_match if best_ratio >= threshold else None

def recursive_json_search(data):
    found_items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            clean_k = clean_key(k)
            if clean_k.lower() not in ["options", "polling", "viability", "candidates", "2024", "statistics", "json", "answer", "reasoning", "analysis", "confidence"]:
                if isinstance(v, (int, float)):
                    found_items[clean_k] = float(v)
                elif isinstance(v, str):
                    try:
                        matches = re.findall(r'\d+\.?\d*', v.replace(",", ""))
                        if matches:
                            found_items[clean_k] = float(matches[-1])
                    except:
                        pass
                if isinstance(v, (dict, list)):
                    found_items.update(recursive_json_search(v))
    elif isinstance(data, list):
        for item in data:
            found_items.update(recursive_json_search(item))
    return found_items

def extract_and_parse(item, logger, context):
    raw_text = ""
    try:
        if hasattr(item, 'results') and isinstance(item.results, dict):
            raw_text = list(item.results.values())[0].llm_response
        elif hasattr(item, 'outputs'):
            raw_text = item.outputs[0].text
        elif isinstance(item, str):
            raw_text = item
    except:
        pass

    if logger and raw_text:
        logger.info(f"--- {context} RAW OUTPUT START ---")
        logger.info(raw_text)
        logger.info(f"--- {context} RAW OUTPUT END ---")

    if not raw_text:
        return {}

    text = re.sub(r'```(?:json)?', r'', raw_text, flags=re.DOTALL)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            try:
                return json.loads(re.sub(r',\s*}', r'}', match.group(0)))
            except:
                pass
    return {}

def detect_and_fix_scale(scores, logger=None, context=""):
    if not scores:
        return scores
    max_val = max(scores.values())
    if 0 < max_val <= 1.0:
        if logger:
            logger.warning(f"{context}: Detected 0-1 scale (max={max_val:.3f}), rescaling to 0-100")
        return {k: v * 100.0 for k, v in scores.items()}
    return scores

def validate_and_log(scores, context, month, logger):
    if not scores:
        logger.warning(f"{context}|{month}: No scores extracted")
        return
    max_score = max(scores.values())
    total = sum(scores.values())
    nonzero = len([v for v in scores.values() if v > 0.5])
    logger.info(f"{context}|{month} Stats: max={max_score:.1f}, total={total:.1f}, nonzero={nonzero}")

def normalize_dict(data, target=100.0):
    total = sum(data.values())
    if total == 0:
        return data
    factor = target / total
    return {k: v * factor for k, v in data.items()}

def format_european_percentage(value):
    return f"{value:.2f}".replace(".", ",")

def reorder_results(df_results, month_order=None):
    if df_results.empty:
        return df_results
    party_order_map = {"Democratic": 0, "Republican": 1, "Libertarian": 2, "Green": 3, "Independent": 4}
    df_results['_p_sort'] = df_results['party'].map(party_order_map)
    if month_order:
        month_map = {m: i for i, m in enumerate(month_order)}
        df_results['_m_sort'] = df_results['month'].map(month_map)
        df_results = df_results.sort_values(['_m_sort', '_p_sort'])
        df_results = df_results.drop(columns=['_p_sort', '_m_sort'])
    else:
        df_results = df_results.sort_values('_p_sort')
        df_results = df_results.drop(columns=['_p_sort'])
    return df_results

def sort_betting_results(df_betting):
    if df_betting.empty:
        return df_betting
    df_betting['_date_sort'] = pd.to_datetime(df_betting['month'], format='%d.%m.%y')
    df_betting['_pct_sort'] = df_betting['poll_percentage'].apply(lambda x: float(str(x).replace(",", ".")))
    df_betting = df_betting.sort_values(['_date_sort', '_pct_sort'], ascending=[True, False])
    df_betting = df_betting.drop(columns=['_date_sort', '_pct_sort'])
    return df_betting

def run_confidence_check(model, logger, options_list, system_prompt, original_stem, context_label, temperature):
    conf_stem = (
        f"You just provided estimates for: {original_stem}\n\n"
        f"Now, for each of the options you just evaluated, explicitly state your CONFIDENCE LEVEL (0-100) "
        f"in the accuracy of that specific prediction.\n"
        f"A score of 100 means you are absolutely certain. A score of 0 means it is a random guess."
    )
    
    rgm_conf = JSONVerbalizedDistribution(
        output_template=(
            "Return ONLY a JSON object where:\n"
            "- Keys are the exact candidate/option names from the list\n"
            "- Values are your CONFIDENCE (0-100) in the prediction you just made\n"
            "- Example: {\"Candidate A\": 85.5, \"Candidate B\": 40.0}\n"
        ),
        output_index_only=False
    )
    
    q_df = pd.DataFrame([{
        "questionnaire_item_id": 99, 
        "question_content": f"Confidence check for {context_label}"
    }])
    
    options = generate_likert_options(
        n=len(options_list), 
        answer_texts=options_list, 
        response_generation_method=rgm_conf, 
        list_prompt_template="Options to rate confidence for:\n{options}"
    )
    
    prompt = LLMPrompt(
        questionnaire_source=q_df, 
        questionnaire_name=f"{context_label}_Confidence", 
        system_prompt=system_prompt,
        prompt=(
            f"{placeholder.PROMPT_OPTIONS}\n\n"
            f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}\n\n"
            f"{placeholder.PROMPT_QUESTIONS}"
        ), 
        seed=None
    )
    
    prompt.prepare_prompt(question_stem=conf_stem, answer_options=options, randomized_item_order=False)
    
    res = conduct_survey_single_item(model, [prompt], max_tokens=512, temperature=temperature)
    raw_json = extract_and_parse(res[0], logger, f"{context_label} (Conf)")
    
    scores = recursive_json_search(raw_json)
    return detect_and_fix_scale(scores, logger, f"{context_label} (Conf)")

CANDIDATES_POOL = {
    "Democratic": ["Kamala Harris", "Joe Biden", "Marianne Williamson", "Jason Palmer", "Dean Phillips", "Robert F. Kennedy Jr."],
    "Republican": ["Donald Trump", "Nikki Haley", "Ron DeSantis", "Asa Hutchinson", "Vivek Ramaswamy", "Chris Christie", "Doug Burgum", "Tim Scott", "Mike Pence", "Larry Elder", "Perry Johnson", "Will Hurd", "Francis Suarez"],
    "Libertarian": ["Charles Ballay", "Jacob Hornberger", "Lars Mapstead", "Chase Oliver", "Michael Rectenwald", "Joshua Smith", "Mike ter Maat"],
    "Green": ["Jill Stein", "Jasmine Sherman", "Jorge Zevala"],
    "Independent": ["Robert F. Kennedy Jr.", "Cornel West"]
}


def run_election_simulation(model, logger, validation_data, finance_data, timeline_by_month, temp_nomination, temp_polling, temp_betting):
    logger.info(f">>> STARTING 2024 ELECTION SIMULATION (FINANCE + TIMELINE + CONFIDENCE) - Temps: nom={temp_nomination}, pol={temp_polling}, bet={temp_betting}")
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November"]
    PARTIES = list(CANDIDATES_POOL.keys())

    poll_targets = validation_data['poll_targets']
    election_target = validation_data['election_target']
    betting_targets = validation_data['betting_targets']
    betting_dates = validation_data['betting_dates']
    date_to_month = validation_data['date_to_month']

    nominees_per_month = {}
    output_hashes = {}
    candidates_with_finance_history = set()

    nomination_results = []
    polling_results = []
    election_results = []
    betting_results = []

    logger.info("--- TASK 1: PARTY NOMINATIONS (WITH FINANCE & TIMELINE) ---")

    previous_nominees = {}

    for month in MONTHS:
        for party in PARTIES:
            base_pool = CANDIDATES_POOL.get(party, [])
            display_pool = base_pool.copy()
            random.shuffle(display_pool)

            # Generate finance descriptions
            finance_descriptions = []
            for cand in display_pool:
                finance_text = get_candidate_finance_text(cand, month, finance_data, candidates_with_finance_history)
                finance_descriptions.append(f"- {finance_text}")

            if not finance_data.empty:
                month_data = finance_data[finance_data['month'] == month]
                if not month_data.empty:
                    candidates_with_finance_history.update(month_data['candidate'].unique())

            finance_context = "\n".join(finance_descriptions)

            # Get timeline events
            timeline_events = format_timeline_events(month, timeline_by_month, max_events=8)
            previous_events = get_cumulative_timeline_summary(month, timeline_by_month, MONTHS)

            # Build question stem with timeline and finance context
            if month in ["January", "February"]:
                stem = (
                    f"TIME PERIOD: {month} 2024 (Early Primary Season) \n\n"
                    f"CONTEXT: Multiple candidates are actively campaigning for the {party} nomination. "
                    f"Primary contests are beginning. Assess each candidate's likelihood of securing the nomination.\n\n"
                )
                if previous_events:
                    stem += (
                        f"CONTEXT FROM PREVIOUS MONTHS:\n"
                        f"{previous_events}\n\n"
                    )
                stem += (
                    f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                    f"{timeline_events}\n\n"
                    f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                    f"{finance_context}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Weight recent campaign events heavily - they show real-world dynamics\n"
                    f"2. Primary results (if in events) directly indicate voter support\n"
                    f"3. Campaign suspensions (shown in events) mean withdrawal - assign <5%\n"
                    f"4. Fundraising indicates viability and momentum\n"
                    f"5. Debates, endorsements, controversies affect standing\n"
                    f"6. Candidates without fundraising data have lower (but not zero) viability\n\n"
                    f"Provide nomination likelihood percentages (0-100) for each candidate."
                )
            elif month in ["March", "April", "May"]:
                stem = (
                    f"TIME PERIOD: {month} 2024 (Mid-Primary Season) \n\n"
                    f"CONTEXT: The {party} nomination race is progressing. Primary contests have been occurring. "
                    f"Some candidates may be gaining or losing momentum. Front-runners may be consolidating support.\n\n"
                )
                if previous_events:
                    stem += (
                        f"CONTEXT FROM PREVIOUS MONTHS:\n"
                        f"{previous_events}\n\n"
                    )
                stem += (
                    f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                    f"{timeline_events}\n\n"
                    f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                    f"{finance_context}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Recent quarterly fundraising shows current momentum\n"
                    f"2. Campaign suspensions (shown in events) = <5% support\n"
                    f"3. Primary wins indicate strong voter support\n"
                    f"4. Front-runners typically consolidate support by this phase\n"
                    f"5. Continued fundraising = viable campaign\n\n"
                    f"Provide nomination likelihood percentages (0-100) for each candidate."
                )
            elif month in ["June", "July", "August"]:
                stem = (
                    f"TIME PERIOD: {month} 2024 (Late Primary / Convention Period) \n\n"
                    f"CONTEXT: The {party} nomination race is in its later stages. By {month}, patterns typically "
                    f"become clearer. One candidate should have dominance (>80% likelihood) if the race is decided.\n\n"
                )
                if previous_events:
                    stem += (
                        f"CONTEXT FROM PREVIOUS MONTHS:\n"
                        f"{previous_events}\n\n"
                    )
                stem += (
                    f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                    f"{timeline_events}\n\n"
                    f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                    f"{finance_context}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. One candidate should have dominance (>80%) if presumptive nominee\n"
                    f"2. Continued fundraising = presumptive nominee\n"
                    f"3. Campaign suspensions = zero viability (<5%)\n"
                    f"4. Convention events (if mentioned) indicate nominee selection\n\n"
                    f"Provide nomination likelihood percentages (0-100) for each candidate."
                )
            else:  # September, October, November
                stem = (
                    f"TIME PERIOD: {month} 2024 (General Election Season) \n\n"
                    f"CONTEXT: The {party} nominee is determined by {month}. The general election campaign is underway. "
                    f"Only the official nominee should have high likelihood (>80%).\n\n"
                )
                if previous_events:
                    stem += (
                        f"CONTEXT FROM PREVIOUS MONTHS:\n"
                        f"{previous_events}\n\n"
                    )
                stem += (
                    f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                    f"{timeline_events}\n\n"
                    f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                    f"{finance_context}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Only the nominee has high likelihood (>80%)\n"
                    f"2. All others have suspended campaigns (<5%)\n"
                    f"3. General election events focus on the nominee\n\n"
                    f"Provide percentages (0-100), with only the nominee receiving significant support."
                )

            rgm_nomination = JSONVerbalizedDistribution(
                output_template=(
                    "Return ONLY a JSON object where:\n"
                    "- Keys are candidate names (exact match from the list)\n"
                    "- Values are decimal numbers between 0 and 100 representing likelihood percentages\n"
                    "- Use 2 decimal places for precision\n"
                    "- No explanatory text, reasoning, or additional fields\n\n"
                    "Example structure (NOT actual values, use your own predictions):\n"
                    '{"Candidate A": XX.XX, "Candidate B": YY.YY, "Candidate C": ZZ.ZZ}\n'
                ),
                output_index_only=False
            )

            # Store stem in DataFrame
            q_df = pd.DataFrame([{
                "questionnaire_item_id": 1, 
                "question_content": f"Nomination likelihood in {month} 2024 for {party} party",
                "question_stem": stem 
            }])

            options = generate_likert_options(
                n=len(display_pool), 
                answer_texts=display_pool, 
                response_generation_method=rgm_nomination, 
                list_prompt_template="Candidates:\n{options}"
            )

            sys_prompt = (
                f"You are analyzing the {party} party nomination race for the 2024 US Presidential Election. "
                f"The current time period is {month} 2024. Use campaign timeline events and finance data to make informed predictions."
            )

            prompt = LLMPrompt(
                questionnaire_source=q_df, 
                questionnaire_name=f"Nomination_{month}_{party}", 
                system_prompt=sys_prompt, 
                prompt=(
                    f"{placeholder.PROMPT_OPTIONS}\n\n"
                    f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}\n\n"
                    f"{placeholder.PROMPT_QUESTIONS}" 
                ), 
                seed=None
            )

            prompt.prepare_prompt(answer_options=options, randomized_item_order=False)

            res = conduct_survey_single_item(model, [prompt], max_tokens=512, temperature=temp_nomination)
            raw_json = extract_and_parse(res[0], logger, f"Nomination {month}|{party}")

            key = f"{party}_nomination"
            if key not in output_hashes:
                output_hashes[key] = set()
            json_str = json.dumps(raw_json, sort_keys=True)
            hash_val = hashlib.md5(json_str.encode()).hexdigest()
            output_hashes[key].add(hash_val)

            scores = recursive_json_search(raw_json)
            scores = detect_and_fix_scale(scores, logger, f"Nomination {month}|{party}")
            
            confs = run_confidence_check(model, logger, display_pool, sys_prompt, stem, f"Nomination {month}|{party}", temp_nomination)
            
            validate_and_log(scores, "Nomination", f"{month}|{party}", logger)

            best_cand, best_score = "Unknown", -1
            for k, v in scores.items():
                match = fuzzy_match_candidate(k, base_pool, threshold=0.6)
                if match and float(v) > best_score:
                    best_score = float(v)
                    best_cand = match
            
            winner_conf = 0.0
            if best_cand != "Unknown":
                if best_cand in confs:
                    winner_conf = confs[best_cand]
                else:
                    for ck, cv in confs.items():
                        if SequenceMatcher(None, best_cand, ck).ratio() > 0.8:
                            winner_conf = cv
                            break

            nominees_per_month[(month, party)] = best_cand
            
            nomination_results.append({
                "month": month,
                "party": party,
                "nominee": best_cand,
                "likelihood_score": format_european_percentage(best_score),
                "confidence": format_european_percentage(winner_conf)
            })

            if party in previous_nominees:
                prev = previous_nominees[party]
                if prev != best_cand:
                    logger.info(f"  [CHANGE] [{month}|{party}] {prev} -> {best_cand}")

            previous_nominees[party] = best_cand
            logger.info(f"  [{month}|{party}] Predicted: {best_cand} ({best_score:.2f}%) [Conf: {winner_conf:.0f}]")

    logger.info("\n--- TASK 2: NATIONAL POLLING & ELECTION (WITH FINANCE & TIMELINE) ---")

    for month in MONTHS:
        roster_labels = []
        party_map = {}
        nominee_finance = []

        for p in PARTIES:
            cand = nominees_per_month.get((month, p), "Unknown")
            if cand == "Unknown": continue
            label = f"{p} (nominee: {cand})"
            roster_labels.append(label)
            party_map[label] = p

            finance_text = get_candidate_finance_text(cand, month, finance_data, candidates_with_finance_history)
            nominee_finance.append(f"- {p} nominee {cand}: {finance_text}")

        random.shuffle(roster_labels)
        finance_context = "\n".join(nominee_finance)

        # Get timeline events
        timeline_events = format_timeline_events(month, timeline_by_month, max_events=8)
        previous_events = get_cumulative_timeline_summary(month, timeline_by_month, MONTHS)

        # Build question stem
        if month == "November":
            sys_prompt = (
                f"You are predicting the ACTUAL ELECTION RESULTS for the 2024 US Presidential Election on November 5, 2024. "
                f"This is the final outcome, not a poll. Use campaign events and finance data to predict actual vote shares."
            )
            stem = (
                f"ELECTION DAY: November 5, 2024 \n\n"
                f"Predict the actual final vote share for each party in the 2024 US Presidential Election. "
                f"This is the real election outcome. Each party has nominated a candidate.\n\n"
            )
            if previous_events:
                stem += (
                    f"CONTEXT FROM PREVIOUS MONTHS:\n"
                    f"{previous_events}\n\n"
                )
            stem += (
                f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                f"{timeline_events}\n\n"
                f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                f"{finance_context}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. This is the ACTUAL election outcome, not a poll\n"
                f"2. Use all available information: events, fundraising, polls\n"
                f"3. Consider voter turnout patterns and election dynamics\n\n"
                f"Provide the final vote share percentages (0-100) for each party."
            )
        else:
            sys_prompt = (
                f"You are predicting national polling for the 2024 US Presidential Election "
                f"as of {month} 2024. Use campaign events and finance data to inform predictions."
            )
            stem = (
                f"POLLING SCENARIO: {month} 2024 National Survey \n\n"
                f"In a general election poll conducted in {month} 2024, what percentage of voters "
                f"would support each party? Each party has nominated a candidate.\n\n"
            )
            if previous_events:
                stem += (
                    f"CONTEXT FROM PREVIOUS MONTHS:\n"
                    f"{previous_events}\n\n"
                )
            stem += (
                f"KEY CAMPAIGN EVENTS IN {month} 2024:\n"
                f"{timeline_events}\n\n"
                f"CAMPAIGN FINANCE DATA AS OF {month} 2024:\n"
                f"{finance_context}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Campaign events show real dynamics and momentum\n"
                f"2. Fundraising indicates campaign strength\n"
                f"3. Consider typical polling patterns for {month}\n\n"
                f"Provide vote share percentages (0-100) for each party."
            )

        rgm_polling = JSONVerbalizedDistribution(
            output_template=(
                "Return a JSON object with vote share predictions BY PARTY:\n"
                'Format: {"Democratic (nominee: Joe Biden)": XX.XX, "Republican (nominee: Donald Trump)": YY.YY, ...}\n'
                "Requirements:\n"
                "- Keys match the party labels provided\n"
                "- Values are percentages (0-100) representing vote share for each party\n"
                "- No explanatory text or additional fields\n"
            ),
            output_index_only=False
        )

        q_df = pd.DataFrame([{
            "questionnaire_item_id": 1, 
            "question_content": f"National party vote share in {month} 2024",
            "question_stem": stem 
        }])

        options = generate_likert_options(
            n=len(roster_labels), 
            answer_texts=roster_labels, 
            response_generation_method=rgm_polling, 
            list_prompt_template="Party tickets:\n{options}"
        )

        prompt = LLMPrompt(
            questionnaire_source=q_df, 
            questionnaire_name=f"Poll_{month}", 
            system_prompt=sys_prompt, 
            prompt=(
                f"{placeholder.PROMPT_OPTIONS}\n\n"
                f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}\n\n"
                f"{placeholder.PROMPT_QUESTIONS}"
            ), 
            seed=None
        )

        prompt.prepare_prompt(answer_options=options, randomized_item_order=False)

        res = conduct_survey_single_item(model, [prompt], max_tokens=512, temperature=temp_polling)
        raw_json = extract_and_parse(res[0], logger, f"Poll {month}")

        scores = recursive_json_search(raw_json)
        scores = detect_and_fix_scale(scores, logger, f"Poll {month}")
        
        confs = run_confidence_check(model, logger, roster_labels, sys_prompt, stem, f"Poll {month}", temp_polling)

        if month == "November":
            target = election_target
        else:
            target = poll_targets.get(month, 100.0)

        scores = normalize_dict(scores, target)

        raw_total = sum(scores.values())
        logger.info(f"  [{month}] Party support sum: {raw_total:.2f}% (target: {target:.2f}%)")

        for k, v in scores.items():
            label_match = fuzzy_match_candidate(k, roster_labels, threshold=0.5)
            if label_match:
                party = party_map[label_match]
                cand = nominees_per_month[(month, party)]
                cand_full = cand
                
                confidence_val = 0.0
                if k in confs:
                    confidence_val = confs[k]
                else:
                    for ck, cv in confs.items():
                        if SequenceMatcher(None, k, ck).ratio() > 0.8:
                            confidence_val = cv
                            break

                if month == "November":
                    election_results.append({
                        'party': party,
                        'candidate': cand_full,
                        'poll_percentage': format_european_percentage(v),
                        'confidence': format_european_percentage(confidence_val)
                    })
                    logger.info(f"  [ELECTION] {party} ({cand_full}): {v:.2f}% [Conf: {confidence_val:.0f}]")
                else:
                    polling_results.append({
                        'month': month, 
                        'party': party, 
                        'candidate': cand_full, 
                        'poll_percentage': format_european_percentage(v),
                        'confidence': format_european_percentage(confidence_val)
                    })
                    logger.info(f"  [{month}] {party} ({cand_full}): {v:.2f}% [Conf: {confidence_val:.0f}]")

    df_poll = reorder_results(pd.DataFrame(polling_results), list(poll_targets.keys()))
    df_election = pd.DataFrame(election_results)
    if not df_election.empty:
        party_order_map = {"Democratic": 0, "Republican": 1, "Libertarian": 2, "Green": 3, "Independent": 4}
        df_election['__sort'] = df_election['party'].map(party_order_map)
        df_election = df_election.sort_values('__sort').drop(columns=['__sort'])
        
    df_nomination = pd.DataFrame(nomination_results)


    logger.info("\n--- TASK 3: PREDICTION MARKETS (WITH FINANCE & TIMELINE) ---")
    betting_results = []

    BETTING_POOL = [
        "Joe Biden", "Donald Trump", "Robert F. Kennedy Jr.", "Kamala Harris", 
        "Ron DeSantis", "Nikki Haley", "Gavin Newsom", "Michelle Obama", 
        "Vivek Ramaswamy", "Mike Pence", "Tim Scott", "Hillary Clinton"
    ]

    for date_str in betting_dates:
        month_name = date_to_month.get(date_str, date_str)

        display_pool = BETTING_POOL.copy()
        random.shuffle(display_pool)

        # Generate finance descriptions
        finance_descriptions = []
        for cand in display_pool:
            finance_text = get_candidate_finance_text(cand, month_name, finance_data, candidates_with_finance_history)
            finance_descriptions.append(f"- {finance_text}")

        finance_context = "\n".join(finance_descriptions)

        # Get timeline events
        timeline_events = format_timeline_events(month_name, timeline_by_month, max_events=8)
        previous_events = get_cumulative_timeline_summary(month_name, timeline_by_month, MONTHS)

        # Build question stem
        stem = (
            f"═══ PREDICTION MARKET: {date_str} ({month_name} 2024) ═══\n\n"
            f"Estimate each candidate's probability of winning the 2024 US Presidential Election "
            f"as would be reflected in prediction markets on {date_str}. "
            f"Markets aggregate various factors including viability, momentum, and general election prospects.\n\n"
        )

        if previous_events:
            stem += (
                f"CONTEXT FROM PREVIOUS MONTHS:\n"
                f"{previous_events}\n\n"
            )

        stem += (
            f"KEY CAMPAIGN EVENTS IN {month_name} 2024:\n"
            f"{timeline_events}\n\n"
            f"CAMPAIGN FINANCE DATA AS OF {month_name} 2024:\n"
            f"{finance_context}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Campaign events directly affect market probabilities\n"
            f"2. Fundraising indicates campaign viability and market sentiment\n"
            f"3. Suspended campaigns = near-zero probability\n"
            f"4. Market probabilities reflect overall win likelihood\n\n"
            f"Provide win probabilities (0-100) for each candidate."
        )

        rgm_betting = JSONVerbalizedDistribution(
            output_template=(
                "Return a JSON object with win probabilities for each candidate:\n"
                'Format: {"Candidate A": XX.XX, "Candidate B": YY.YY, ...}\n'
                "Requirements:\n"
                "- Keys are candidate names from the provided list\n"
                "- Values represent probability (0-100) of winning the 2024 presidential election\n"
                "- No explanatory text or additional fields\n"
            ),
            output_index_only=False
        )

        q_df = pd.DataFrame([{
            "questionnaire_item_id": 1, 
            "question_content": f"Prediction market probabilities on {date_str}",
            "question_stem": stem 
        }])

        options = generate_likert_options(
            n=len(display_pool), 
            answer_texts=display_pool, 
            response_generation_method=rgm_betting, 
            list_prompt_template="Market candidates:\n{options}"
        )

        sys_prompt = (
            f"You are estimating prediction market probabilities for the 2024 US Presidential Election as of {date_str}. "
            f"Use campaign events and finance data to inform market probability estimates."
        )

        prompt = LLMPrompt(
            questionnaire_source=q_df, 
            questionnaire_name=f"Betting_{date_str}",
            system_prompt=sys_prompt, 
            prompt=(
                f"{placeholder.PROMPT_OPTIONS}\n\n"
                f"{placeholder.PROMPT_AUTOMATIC_OUTPUT_INSTRUCTIONS}\n\n"
                f"{placeholder.PROMPT_QUESTIONS}"
            ), 
            seed=None
        )

        prompt.prepare_prompt(answer_options=options, randomized_item_order=False)

        res = conduct_survey_single_item(model, [prompt], max_tokens=512, temperature=temp_betting)
        raw_json = extract_and_parse(res[0], logger, f"Betting {date_str}")

        scores = recursive_json_search(raw_json)
        scores = detect_and_fix_scale(scores, logger, f"Betting {date_str}")
        
        confs = run_confidence_check(model, logger, display_pool, sys_prompt, stem, f"Betting {date_str}", temp_betting)

        sorted_bets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_bets[:5]

        target = betting_targets.get(date_str, 100.0)
        top_5_dict = {k: v for k, v in top_5}
        top_5_normalized = normalize_dict(top_5_dict, target)

        logger.info(f"  [{date_str} ({month_name})] Top 5 candidates (sum: {target:.2f}%):")

        for name, score in sorted(top_5_normalized.items(), key=lambda x: x[1], reverse=True):
            match = fuzzy_match_candidate(name, BETTING_POOL, threshold=0.6)
            if not match:
                match = name
            
            confidence_val = 0.0
            best_match_score = 0
            for ck, cv in confs.items():
                ratio = SequenceMatcher(None, name, ck).ratio()
                if ratio > 0.8 and ratio > best_match_score:
                    best_match_score = ratio
                    confidence_val = cv

            betting_results.append({
                'month': date_str,
                'candidate': match, 
                'poll_percentage': format_european_percentage(score),
                'confidence': format_european_percentage(confidence_val)
            })
            logger.info(f"    {match}: {score:.2f}% [Conf: {confidence_val:.0f}]")

    df_bet = pd.DataFrame(betting_results)
    df_bet = sort_betting_results(df_bet)

    return df_poll, df_election, df_bet, df_nomination


def main():
    parser_args = argparse.ArgumentParser()
    parser_args.add_argument("--model_id", type=str, required=True)
    parser_args.add_argument("--model_name", type=str, required=True)
    parser_args.add_argument("--tp_size", type=int, default=1)
    args = parser_args.parse_args()

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    unique_id = f"{job_id}_{task_id}"

    output_base_dir = "outputs"
    model_output_dir = os.path.join(output_base_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger = setup_logging(args.model_name, unique_id)
    logger.info(f"Output Directory: {model_output_dir}")

    try:
        random.seed(42)
        logger.info("Random seed=42 for reproducibility")

        logger.info("\n--- LOADING VALIDATION DATA ---")
        validation_data = load_validation_data(logger)

        logger.info("\n--- LOADING CAMPAIGN FINANCE DATA ---")
        finance_data = load_finance_data(logger)

        logger.info("\n--- LOADING CAMPAIGN TIMELINE DATA ---")
        timeline_by_month = load_timeline_data(logger)

        # Define 5 temperature configurations
        temp_betting = 0.3  # Fixed
        temp_configs = [
            (0.1, 0.1, temp_betting),  # T_nom=0.1, T_poll=0.1
            (0.3, 0.7, temp_betting),  # T_nom=0.3, T_poll=0.7
            (0.5, 0.5, temp_betting),  # T_nom=0.5, T_poll=0.5
            (0.7, 0.3, temp_betting),  # T_nom=0.7, T_poll=0.3
            (0.9, 0.9, temp_betting),  # T_nom=0.9, T_poll=0.9  
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Two-Stage Temperature Optimization Experiment")
        logger.info(f"{'='*60}")
        logger.info(f"  Betting temperature fixed at: {temp_betting}")
        logger.info(f"  Testing {len(temp_configs)} (Nomination, Polling) combinations:")
        logger.info(f"")
        for i, (t_nom, t_pol, t_bet) in enumerate(temp_configs):
            logger.info(f"    Config {i+1}: Nom={t_nom}, Poll={t_pol}, Bet={t_bet}")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"Initializing {args.model_name}...")

        model = LLM(
            model=args.model_id, 
            max_model_len=2048, 
            seed=42,
            trust_remote_code=True, 
            dtype="bfloat16", 
            tensor_parallel_size=args.tp_size, 
            gpu_memory_utilization=0.92,
            enforce_eager=True, 
            disable_log_stats=True
        )
        
        # Run simulation for each temperature configuration
        for config_idx, (t_nom, t_pol, t_bet) in enumerate(temp_configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING CONFIG {config_idx + 1}/{len(temp_configs)}")
            logger.info(f"  Nomination temp: {t_nom}")
            logger.info(f"  Polling temp:    {t_pol}")
            logger.info(f"  Betting temp:    {t_bet}")
            logger.info(f"{'='*60}\n")

            df_poll, df_election, df_bet, df_nomination = run_election_simulation(
                model, logger, validation_data, finance_data, timeline_by_month,
                temp_nomination=t_nom,
                temp_polling=t_pol,
                temp_betting=t_bet
            )

            # Generate temperature code for filename
            # Format: n{nom}p{poll} 
            # Example: n3p7 means nom=0.3, poll=0.7
            temp_code = f"n{int(t_nom*10)}p{int(t_pol*10)}"

            if not df_poll.empty:
                fname = os.path.join(model_output_dir, f"national_polls_results_{args.model_name}_finance_timeline_{temp_code}_{unique_id}.csv")
                df_poll.to_csv(fname, sep=";", index=False, encoding='utf-8-sig')
                logger.info(f"✓ Saved Polling: {fname}")

            if not df_election.empty:
                fname = os.path.join(model_output_dir, f"national_election_results_{args.model_name}_finance_timeline_{temp_code}_{unique_id}.csv")
                df_election.to_csv(fname, sep=";", index=False, encoding='utf-8-sig')
                logger.info(f"✓ Saved Election: {fname}")

            if not df_bet.empty:
                fname = os.path.join(model_output_dir, f"national_betting_odds_results_{args.model_name}_finance_timeline_{temp_code}_{unique_id}.csv")
                df_bet.to_csv(fname, sep=";", index=False, encoding='utf-8-sig')
                logger.info(f"✓ Saved Betting: {fname}")
                
            if not df_nomination.empty:
                fname = os.path.join(model_output_dir, f"nomination_confidence_{args.model_name}_finance_timeline_{temp_code}_{unique_id}.csv")
                df_nomination.to_csv(fname, sep=";", index=False, encoding='utf-8-sig')
                logger.info(f"✓ Saved Nomination Confidence: {fname}")
                
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ COMPLETED ALL {len(temp_configs)} TEMPERATURE CONFIGURATIONS")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"FATAL: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()