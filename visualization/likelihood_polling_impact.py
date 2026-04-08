"""
Nomination Likelihood Score → Prediction quality of Polling
=============================================

X：Nomination Likelihood Score（box：low/middle/high）
Y：Polling-MAE / Pearson
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import quick_load

# ============================================================================
# configuration
# ============================================================================

OUTPUT_FILE = "likelihood_polling_impact.png"

# ============================================================================
# main function
# ============================================================================

def main():
    print("loading...")
    df_polling, df_nomination, df_metrics = quick_load(verbose=False)
    
    if df_nomination.empty or df_polling.empty:
        print("✗ not enough data")
        return
    
    # Check required columns
    if 'likelihood_score' not in df_nomination.columns:
        print("✗ nomination does not have likelihood_score column")
        return
    
    print(f"Nomination: {len(df_nomination)} rows")
    print(f"Polling: {len(df_polling)} rows")
    
    # ========================================================================
    # Merge Data: Correlate nomination likelihood with polling error
    # Correlation keys: model, level, temp_config, month, party
    # ========================================================================
    
    # Nomination：Compute the average likelihood for each combination.
    nom_agg = df_nomination.groupby(['model', 'level', 'temp_config', 'month', 'party']).agg({
        'likelihood_score': 'mean'
    }).reset_index()
    
    # Polling：Calculate error
    if 'actual' not in df_polling.columns or df_polling['actual'].isna().all():
        print("✗ polling does not have actual column or all values are missing")
        return
    
    df_poll_valid = df_polling[df_polling['actual'].notna()].copy()
    df_poll_valid['abs_error'] = (df_poll_valid['poll_percentage'] - df_poll_valid['actual']).abs()
    
    
    merged = df_poll_valid.merge(
        nom_agg,
        on=['model', 'level', 'temp_config', 'month', 'party'],
        how='inner'
    )
    
    print(f"Merged: {len(merged)} rows")
    
    if merged.empty:
        print("✗ Merged data is empty")
        return
    
    # ========================================================================
    # Bucket the likelihood_score 
    # ========================================================================
    
    bins = [0, 50, 75, 90, 100]
    labels = ['Low\n(0-50)', 'Medium\n(50-75)', 'High\n(75-90)', 'Very High\n(90-100)']
    merged['likelihood_bin'] = pd.cut(merged['likelihood_score'], bins=bins, labels=labels)
    
    # Clustering by Box
    agg = merged.groupby('likelihood_bin').agg({
        'abs_error': ['mean', 'std', 'count']
    }).reset_index()
    agg.columns = ['likelihood_bin', 'MAE', 'MAE_std', 'count']
    
    # Compute Pearson (grouped by box)
    pearson_list = []
    for bin_label in labels:
        df_bin = merged[merged['likelihood_bin'] == bin_label]
        if len(df_bin) > 2:
            r = df_bin['poll_percentage'].corr(df_bin['actual'])
            pearson_list.append({'likelihood_bin': bin_label, 'Pearson_r': r})
        else:
            pearson_list.append({'likelihood_bin': bin_label, 'Pearson_r': np.nan})
    
    pearson_df = pd.DataFrame(pearson_list)
    agg = agg.merge(pearson_df, on='likelihood_bin')
    
    print("\nAggregated Results:")
    print(agg.to_string(index=False))
    
    # ========================================================================
    # Image
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    # MAE
    ax1 = axes[0]
    x = range(len(agg))
    bars1 = ax1.bar(x, agg['MAE'], yerr=agg['MAE_std'], color=colors, 
                   alpha=0.8, edgecolor='black', capsize=5)
    
    for bar, mae, cnt in zip(bars1, agg['MAE'], agg['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{mae:.2f}\n(n={cnt})', ha='center', fontsize=9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg['likelihood_bin'])
    ax1.set_xlabel('Nomination Likelihood Score', fontsize=11)
    ax1.set_ylabel('Polling MAE (↓ lower is better)', fontsize=11)
    ax1.set_title('A) Polling MAE by Nomination Likelihood', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Pearson
    ax2 = axes[1]
    bars2 = ax2.bar(x, agg['Pearson_r'], color=colors, alpha=0.8, edgecolor='black')
    
    for bar, r in zip(bars2, agg['Pearson_r']):
        if pd.notna(r):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r:.3f}', ha='center', fontsize=10)
    
    ax2.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='r=0.8')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agg['likelihood_bin'])
    ax2.set_xlabel('Nomination Likelihood Score', fontsize=11)
    ax2.set_ylabel('Polling Pearson r (↑ higher is better)', fontsize=11)
    ax2.set_title('B) Polling Pearson r by Nomination Likelihood', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Impact of Nomination Likelihood on Polling Prediction Quality\n'
                 '(Higher likelihood = more confident nomination prediction)',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
