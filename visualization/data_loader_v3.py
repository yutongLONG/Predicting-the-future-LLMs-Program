

import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Constants
# ============================================================================

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November"]

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
    "joe": "Joe Biden"
}

MODEL_MAP = {
    "Llama-3.1-8B": "Llama-8B",
    "Llama-3.3-70B": "Llama-70B",
    "Mistral-7B": "Mistral-7B",
    "Mixtral-8x7B": "Mixtral-8x7B"
}

TEMP_CONFIG_MAP = {
    'n1p1': (0.1, 0.1),
    'n3p7': (0.3, 0.7),
    'n5p5': (0.5, 0.5),
    'n7p3': (0.7, 0.3),
    'n9p9': (0.9, 0.9)
}

TEMP_ORDER = ['n1p1', 'n3p7', 'n5p5', 'n7p3', 'n9p9']
LEVEL_ORDER = ['Basic', 'Finance', 'Timeline', 'Finance+Timeline']


# ============================================================================
# Helper Functions
# ============================================================================

def standardize_name(name: str) -> str:
    if pd.isna(name):
        return name
    name_lower = str(name).strip().lower()
    return NAME_MAP.get(name_lower, str(name).strip())


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.'), 
        errors='coerce'
    )


def extract_metadata_from_filename(filepath: str) -> Dict:
    filename = os.path.basename(filepath)
    metadata = {}
    
    # Task type
    if 'polls' in filename or 'election' in filename:
        metadata['task_type'] = 'polling'
    elif 'nomination' in filename:
        metadata['task_type'] = 'nomination'
    else:
        metadata['task_type'] = 'unknown'
    
    # Model
    for model_raw, model_short in MODEL_MAP.items():
        if model_raw in filename:
            metadata['model'] = model_short
            break
    else:
        metadata['model'] = 'Unknown'
    
    # Temp config
    temp_match = re.search(r'_(n\d+p\d+)_', filename)
    if temp_match:
        metadata['temp_config'] = temp_match.group(1)
    else:
        metadata['temp_config'] = None
    
    # Level
    fname_lower = filename.lower()
    if 'finance_timeline' in fname_lower:
        metadata['level'] = 'Finance+Timeline'
    elif 'timeline' in fname_lower:
        metadata['level'] = 'Timeline'
    elif 'finance' in fname_lower:
        metadata['level'] = 'Finance'
    else:
        metadata['level'] = 'Basic'
    
    return metadata


# ============================================================================
# Main Loader
# ============================================================================

class DataLoader:
    def __init__(self, outputs_dir: str = None, validation_dir: str = None, 
                 cache_dir: str = None, verbose: bool = True):
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.outputs_dir = outputs_dir or os.path.join(script_dir, 'outputs')
        self.validation_dir = validation_dir or os.path.join(script_dir, 'validation')
        self.cache_dir = cache_dir or os.path.join(script_dir, '.cache')
        self.verbose = verbose
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.df_validation = None
    
    def _log(self, msg):
        if self.verbose:
            print(msg)
    
    def _get_cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f'{name}.pkl')
    
    def _load_cache(self, name: str):
        path = self._get_cache_path(name)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self._log(f"✓ load from cache {name}")
                return data
            except:
                pass
        return None
    
    def _save_cache(self, name: str, data):
        path = self._get_cache_path(name)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self._log(f"✓ cache saved {name}")
        except Exception as e:
            self._log(f"✗ cache save failed: {e}")
    
    def load_validation(self):
        
        path = os.path.join(self.validation_dir, 'national_polls_results.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
            df['poll_percentage'] = clean_numeric(df['poll_percentage'])
            if 'candidate' in df.columns:
                df['candidate'] = df['candidate'].apply(standardize_name)
            self.df_validation = df
            self._log(f"✓ validation set: {len(df)} rows")
        else:
            self._log("✗ validation set not found")
    
    def load_polling(self, use_cache: bool = True) -> pd.DataFrame:
       
        
        if use_cache:
            cached = self._load_cache('polling')
            if cached is not None:
                return cached
        
        self._log("\nloading Polling...")
        start = time.time()
        
        all_data = []
        file_count = 0
        
        for root, dirs, files in os.walk(self.outputs_dir):
            for fname in files:
                if not fname.endswith('.csv'):
                    continue
                if 'polls' not in fname and 'election' not in fname:
                    continue
                if '_n' not in fname:
                    continue
                
                fpath = os.path.join(root, fname)
                meta = extract_metadata_from_filename(fpath)
                
                if meta['temp_config'] is None:
                    continue
                
                try:
                    df = pd.read_csv(fpath, sep=';', encoding='utf-8-sig')
                    df['poll_percentage'] = clean_numeric(df['poll_percentage'])
                    if 'confidence' in df.columns:
                        df['confidence'] = clean_numeric(df['confidence'])
                    if 'candidate' in df.columns:
                        df['candidate'] = df['candidate'].apply(standardize_name)
                    
                    df['model'] = meta['model']
                    df['level'] = meta['level']
                    df['temp_config'] = meta['temp_config']
                    
                    all_data.append(df)
                    file_count += 1
                except:
                    pass
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            elapsed = time.time() - start
            self._log(f"✓ Polling: {file_count} 文件, {len(result)} 行 ({elapsed:.1f}秒)")
            
            if use_cache:
                self._save_cache('polling', result)
            
            return result
        
        return pd.DataFrame()
    
    def load_nomination(self, use_cache: bool = True) -> pd.DataFrame:
        """loading all nomination"""
        
        if use_cache:
            cached = self._load_cache('nomination')
            if cached is not None:
                return cached
        
        self._log("\nloading Nomination")
        start = time.time()
        
        all_data = []
        file_count = 0
        
        for root, dirs, files in os.walk(self.outputs_dir):
            for fname in files:
                if not fname.endswith('.csv'):
                    continue
                if 'nomination' not in fname:
                    continue
                if '_n' not in fname:
                    continue
                
                fpath = os.path.join(root, fname)
                meta = extract_metadata_from_filename(fpath)
                
                if meta['temp_config'] is None:
                    continue
                
                try:
                    df = pd.read_csv(fpath, sep=';', encoding='utf-8-sig')
                    if 'likelihood_score' in df.columns:
                        df['likelihood_score'] = clean_numeric(df['likelihood_score'])
                    if 'confidence' in df.columns:
                        df['confidence'] = clean_numeric(df['confidence'])
                    
                    df['model'] = meta['model']
                    df['level'] = meta['level']
                    df['temp_config'] = meta['temp_config']
                    
                    all_data.append(df)
                    file_count += 1
                except:
                    pass
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            elapsed = time.time() - start
            self._log(f"Nomination: {file_count} files, {len(result)} rows ({elapsed:.1f}s)")
            
            if use_cache:
                self._save_cache('nomination', result)
            
            return result
        
        return pd.DataFrame()
    
    def join_validation(self, df_polling: pd.DataFrame) -> pd.DataFrame:
        """merge validation set, calculate error"""
        if df_polling.empty or self.df_validation is None:
            return df_polling
        
        df = df_polling.merge(
            self.df_validation[['month', 'party', 'candidate', 'poll_percentage']],
            on=['month', 'party', 'candidate'],
            how='left',
            suffixes=('', '_actual')
        )
        df.rename(columns={'poll_percentage_actual': 'actual'}, inplace=True)
        df['error'] = df['poll_percentage'] - df['actual']
        df['abs_error'] = df['error'].abs()
        
        matched = df['actual'].notna().sum()
        self._log(f"validation matched: {matched}/{len(df)}")
        
        return df
    
    def calculate_metrics(self, df_polling: pd.DataFrame) -> pd.DataFrame:
        """calculate aggregated metrics"""
        df_valid = df_polling[df_polling['actual'].notna()].copy()
        
        if df_valid.empty:
            return pd.DataFrame()
        
        metrics = []
        for (model, level, temp_config), group in df_valid.groupby(['model', 'level', 'temp_config']):
            mae = group['abs_error'].mean()
            mse = (group['error'] ** 2).mean()
            rmse = np.sqrt(mse)
            
            try:
                r, p = pearsonr(group['poll_percentage'], group['actual']) if len(group) > 2 else (np.nan, np.nan)
            except:
                r, p = np.nan, np.nan
            
            metrics.append({
                'model': model,
                'level': level,
                'temp_config': temp_config,
                'MAE': mae,
                'RMSE': rmse,
                'Pearson_r': r,
                'Pearson_p': p,
                'count': len(group)
            })
        
        return pd.DataFrame(metrics)
    
    def load_all(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        self._log("=" * 50)
        self._log("Data Loader")
        self._log("=" * 50)
        
       
        self.load_validation()
        
        # Polling
        df_polling = self.load_polling(use_cache)
        if not df_polling.empty:
            df_polling = self.join_validation(df_polling)
        
        # Nomination
        df_nomination = self.load_nomination(use_cache)
        
        # Metrics
        df_metrics = self.calculate_metrics(df_polling) if not df_polling.empty else pd.DataFrame()
        
        self._log("\n" + "=" * 50)
        self._log(f"Complete: Polling {len(df_polling)} rows, Nomination {len(df_nomination)} rows")
        self._log("=" * 50)
        
        return df_polling, df_nomination, df_metrics


# ============================================================================
# Convenience Function
# ============================================================================

def quick_load(outputs_dir: str = None, validation_dir: str = None, 
               use_cache: bool = True, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    loader = DataLoader(outputs_dir=outputs_dir, validation_dir=validation_dir, verbose=verbose)
    return loader.load_all(use_cache=use_cache)


def clear_cache(cache_dir: str = None):
    if cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, '.cache')
    
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"✓ cleared: {cache_dir}")


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    df_polling, df_nomination, df_metrics = quick_load()
    
   
    
    if not df_polling.empty:
        print(f"\nPolling columns: {df_polling.columns.tolist()}")
        print(df_polling.head(3))
    
    if not df_nomination.empty:
        print(f"\nNomination columns: {df_nomination.columns.tolist()}")
        print(df_nomination.head(3))
    
    if not df_metrics.empty:
        print(f"\nMetrics:")
        print(df_metrics.head())
