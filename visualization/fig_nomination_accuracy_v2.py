

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_loader_v3 import quick_load, TEMP_ORDER, LEVEL_ORDER, MONTH_ORDER

OUTPUT_FILE = "fig_nomination_accuracy_v2.png"

MODEL_ORDER = ['Llama-8B', 'Llama-70B', 'Mistral-7B', 'Mixtral-8x7B']
PARTY_ORDER = ['Democratic', 'Republican', 'Green', 'Libertarian', 'Independent']

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


def load_ground_truth(validation_dir=None):
    """Load ground truth: candidate for each (month, party)"""
    if validation_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        validation_dir = os.path.join(script_dir, 'validation')
    
    path = os.path.join(validation_dir, 'national_polls_results.csv')
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    
    # Standardize names
    df['candidate'] = df['candidate'].apply(standardize_name)
    df = df.rename(columns={'candidate': 'true_candidate'})
    
    return df[['month', 'party', 'true_candidate']]


def calculate_accuracy(df_polling, df_ground_truth):
    """Calculate accuracy for each (model, level, temp_config, party)"""
    
    df_poll = df_polling.copy()
    
    # Standardize names
    df_poll['candidate'] = df_poll['candidate'].apply(standardize_name)
    
    # Merge ground truth
    df_merged = df_poll.merge(df_ground_truth, on=['month', 'party'], how='left')
    
    # Compare standardized names
    df_merged['correct'] = df_merged['candidate'] == df_merged['true_candidate']
    
    # Calculate accuracy per (model, level, temp_config, party)
    accuracy = df_merged.groupby(['model', 'level', 'temp_config', 'party']).agg({
        'correct': 'mean'
    }).reset_index()
    accuracy.rename(columns={'correct': 'accuracy'}, inplace=True)
    
    return accuracy


def main():
    print("Loading data...")
    df_polling, df_nomination, df_metrics = quick_load(verbose=False)
    
    if df_polling.empty:
        print("No polling data")
        return
    
    print("Loading ground truth...")
    df_gt = load_ground_truth()
    
    print("Calculating accuracy (fuzzy match)...")
    df_acc = calculate_accuracy(df_polling, df_gt)
    
    models = [m for m in MODEL_ORDER if m in df_acc['model'].unique()]
    levels = [l for l in LEVEL_ORDER if l in df_acc['level'].unique()]
    
    n_rows = len(models)
    n_cols = len(levels)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows), squeeze=False)
    
    for i, model in enumerate(models):
        for j, level in enumerate(levels):
            ax = axes[i, j]
            
            df_sub = df_acc[(df_acc['model'] == model) & (df_acc['level'] == level)]
            
            if df_sub.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model} | {level}', fontsize=9)
                continue
            
            # Pivot: row=party, col=temp_config, value=accuracy
            pivot = df_sub.pivot(index='party', columns='temp_config', values='accuracy')
            
            parties_present = [p for p in PARTY_ORDER if p in pivot.index]
            temps_present = [t for t in TEMP_ORDER if t in pivot.columns]
            pivot = pivot.reindex(index=parties_present, columns=temps_present)
            
            # Heatmap
            sns.heatmap(pivot, ax=ax, cmap='RdYlGn', 
                       annot=True, fmt='.0%', annot_kws={'size': 8},
                       cbar=False, linewidths=0.5, linecolor='white',
                       vmin=0, vmax=1)
            
            ax.set_title(f'{model} | {level}', fontsize=10, fontweight='bold')
            ax.set_xlabel('temp_config' if i == n_rows - 1 else '')
            ax.set_ylabel('')
            ax.tick_params(labelsize=8)
    
    # Overall accuracy
    overall_acc = df_acc['accuracy'].mean() * 100
    
    plt.suptitle(f'Nomination Accuracy: Polling Candidate vs Ground Truth\n'
                 f'(Overall: {overall_acc:.1f}%)',
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_FILE}")
    print(f"\nOverall accuracy: {overall_acc:.1f}%")


if __name__ == '__main__':
    main()
