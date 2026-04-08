

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_loader_v3 import quick_load, TEMP_ORDER, LEVEL_ORDER

OUTPUT_FILE = "fig1_tempconfig_mae_line_v2.png"

LEVEL_COLORS = {
    'Basic': '#e74c3c',
    'Finance': '#3498db',
    'Timeline': '#2ecc71',
    'Finance+Timeline': '#9b59b6'
}

MODEL_ORDER = ['Llama-8B', 'Llama-70B', 'Mistral-7B', 'Mixtral-8x7B']

# Name standardization map
NAME_MAP = {
    "trump": "Donald Trump",
    "donald trump": "Donald Trump",
    "biden": "Joe Biden",
    "joe biden": "Joe Biden",
    "harris": "Kamala Harris",
    "kamala harris": "Kamala Harris",
    "kamala": "Kamala Harris",
    "kennedy": "Robert F. Kennedy Jr.",
    "rfk": "Robert F. Kennedy Jr.",
    "robert f. kennedy jr.": "Robert F. Kennedy Jr.",
    "robert kennedy": "Robert F. Kennedy Jr.",
    "joe": "Joe Biden",
    "stein": "Jill Stein",
    "jill stein": "Jill Stein",
    "oliver": "Chase Oliver",
    "chase oliver": "Chase Oliver",
}


def standardize_name(name):
    """Standardize candidate name using NAME_MAP"""
    if pd.isna(name):
        return name
    name_lower = str(name).lower().strip()
    return NAME_MAP.get(name_lower, str(name).strip())


def clean_numeric(val):
    """Convert European format to float"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).replace(',', '.'))


def load_ground_truth(validation_dir=None):
    """Load ground truth: (month, party, candidate) -> poll_percentage"""
    if validation_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        validation_dir = os.path.join(script_dir, 'validation')
    
    path = os.path.join(validation_dir, 'national_polls_results.csv')
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    
    df['poll_percentage'] = df['poll_percentage'].apply(clean_numeric)
    # Standardize names
    df['candidate'] = df['candidate'].apply(standardize_name)
    df = df.rename(columns={'candidate': 'true_candidate', 'poll_percentage': 'actual'})
    
    return df[['month', 'party', 'true_candidate', 'actual']]


def calculate_mae(df_polling, df_ground_truth):
    """Calculate MAE using name standardization"""
    
    df_poll = df_polling.copy()
    df_gt = df_ground_truth.copy()
    
    # Ensure poll_percentage is numeric
    df_poll['poll_percentage'] = df_poll['poll_percentage'].apply(clean_numeric)
    
    # Standardize names
    df_poll['candidate'] = df_poll['candidate'].apply(standardize_name)
    
    # Debug: print unique candidates
    print(f"  Polling candidates: {df_poll['candidate'].unique()[:10]}")
    print(f"  Ground truth candidates: {df_gt['true_candidate'].unique()[:10]}")
    
    # Merge on month, party, and standardized candidate name
    df_merged = df_poll.merge(
        df_gt, 
        left_on=['month', 'party', 'candidate'],
        right_on=['month', 'party', 'true_candidate'],
        how='inner'
    )
    
    print(f"  Matched rows: {len(df_merged)}")
    print(f"  Columns after merge: {df_merged.columns.tolist()}")
    
    if df_merged.empty:
        print("  Warning: No matches found!")
        return pd.DataFrame()
    
    # Calculate absolute error (actual_y is from ground truth)
    df_merged['abs_error'] = (df_merged['poll_percentage'] - df_merged['actual_y']).abs()
    
    # Calculate MAE per (model, level, temp_config)
    mae = df_merged.groupby(['model', 'level', 'temp_config']).agg({
        'abs_error': 'mean'
    }).reset_index()
    mae.rename(columns={'abs_error': 'MAE'}, inplace=True)
    
    return mae


def main():
    print("Loading data...")
    df_polling, df_nomination, df_metrics = quick_load(verbose=False)
    
    if df_polling.empty:
        print("No polling data")
        return
    
    print("Loading ground truth...")
    df_gt = load_ground_truth()
    
    print("Calculating MAE...")
    df_mae = calculate_mae(df_polling, df_gt)
    
    if df_mae.empty:
        print("No MAE data after matching")
        return
    
    models = [m for m in MODEL_ORDER if m in df_mae['model'].unique()]
    levels = [l for l in LEVEL_ORDER if l in df_mae['level'].unique()]
    
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 5), squeeze=False)
    axes = axes[0]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        df_model = df_mae[df_mae['model'] == model]
        
        for level in levels:
            df_level = df_model[df_model['level'] == level]
            
            if df_level.empty:
                continue
            
            # Reorder by temp_config
            df_level = df_level.set_index('temp_config').reindex(TEMP_ORDER)
            
            ax.plot(range(len(TEMP_ORDER)), df_level['MAE'].values,
                   'o-', color=LEVEL_COLORS.get(level, 'gray'),
                   linewidth=2, markersize=6, label=level)
        
        ax.set_xticks(range(len(TEMP_ORDER)))
        ax.set_xticklabels(TEMP_ORDER)
        ax.set_xlabel('Temperature Config')
        ax.set_ylabel('MAE' if idx == 0 else '')
        ax.set_title(model, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if idx == len(models) - 1:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Figure 1: Polling MAE by Temperature Config\n(per model × level × config)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()