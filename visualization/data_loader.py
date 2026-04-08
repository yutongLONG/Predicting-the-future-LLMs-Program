

import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Constants
# ============================================================================

# Month order
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November"]

# Candidate name standardization mapping
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

# Model name mapping (simplified display)
MODEL_MAP = {
    "Llama-3.1-8B": "Llama-8B",
    "Llama-3.3-70B": "Llama-70B",
    "Mistral-7B": "Mistral-7B",
    "Mixtral-8x7B": "Mixtral-8x7B"
}

# Temperature configuration mapping
# Format: n{nom}p{poll} -> (nomination_temp, polling_temp)
# betting_temp is FIXED
TEMP_CONFIG_MAP = {
    'n1p1': (0.1, 0.1),   # T_nom=0.1, T_poll=0.1
    'n3p7': (0.3, 0.7),   # T_nom=0.3, T_poll=0.7
    'n5p5': (0.5, 0.5),   # T_nom=0.5, T_poll=0.5
    'n7p3': (0.7, 0.3),   # T_nom=0.7, T_poll=0.3
    'n9p9': (0.9, 0.9)    # T_nom=0.9, T_poll=0.9
}


# ============================================================================
# Helper Functions
# ============================================================================

def standardize_name(name: str) -> str:
    """
    Standardize candidate names
    
    Purpose: Unify candidate names between validation and prediction sets
    Example: 'Biden' -> 'Joe Biden', 'Trump' -> 'Donald Trump'
    """
    if pd.isna(name):
        return name
    name_lower = str(name).strip().lower()
    return NAME_MAP.get(name_lower, str(name).strip())


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Clean numeric format (European format -> US format)
    
    Purpose: Convert '45,23' to 45.23
    """
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.'), 
        errors='coerce'
    )


def extract_metadata_from_filename(filepath: str) -> Dict:
    """
    Extract metadata from filename
    
    """
    filename = os.path.basename(filepath)
    metadata = {}
    
    # 1. Extract task type
    if 'polls' in filename or 'election' in filename:
        metadata['task_type'] = 'polling'
    elif 'betting' in filename:
        metadata['task_type'] = 'betting'
    elif 'nomination' in filename:
        metadata['task_type'] = 'nomination'
    else:
        metadata['task_type'] = 'unknown'
    
    # 2. Extract model name
    for model_raw, model_short in MODEL_MAP.items():
        if model_raw in filename:
            metadata['model_raw'] = model_raw
            metadata['model'] = model_short
            break
    else:
        metadata['model_raw'] = 'Unknown'
        metadata['model'] = 'Unknown'
    
    # 3. Extract temperature configuration
    # Format: n{nom}p{poll} (e.g., n3p7, n5p5)
    temp_match = re.search(r'_(n\d+p\d+)_', filename)
    if temp_match:
        temp_config = temp_match.group(1)
        metadata['temp_config'] = temp_config
        
        # Temperature Mapping
        if temp_config in TEMP_CONFIG_MAP:
            nomination_temp, polling_temp = TEMP_CONFIG_MAP[temp_config]
            metadata['nomination_temp'] = nomination_temp
            metadata['polling_temp'] = polling_temp
        else:
            metadata['nomination_temp'] = None
            metadata['polling_temp'] = None
    else:
        metadata['temp_config'] = None
        metadata['nomination_temp'] = None
        metadata['polling_temp'] = None
    
    # 4. Extract information level
    fname_lower = filename.lower()
    if 'finance_timeline' in fname_lower:
        metadata['level'] = 'Finance+Timeline'
    elif 'timeline' in fname_lower:
        metadata['level'] = 'Timeline'
    elif 'finance' in fname_lower:
        metadata['level'] = 'Finance'
    else:
        metadata['level'] = 'Basic'
    
    # 5. Extract job_id and task_id
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            metadata['job_id'] = parts[-2]
            metadata['task_id'] = parts[-1].replace('.csv', '')
        except:
            metadata['job_id'] = 'unknown'
            metadata['task_id'] = 'unknown'
    
    return metadata


# ============================================================================
# Main Loader Class
# ============================================================================

class TemperatureExperimentLoader:
    """
    Temperature Scan Experiment Data Loader

    """
    
    def __init__(self, outputs_dir: str = None, 
                 validation_dir: str = None,
                 verbose: bool = True):
        """
        Initialize
    
        """
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If no path specified, use relative path from script directory
        if outputs_dir is None:
            outputs_dir = os.path.join(script_dir, 'outputs')
        if validation_dir is None:
            validation_dir = os.path.join(script_dir, 'validation')
        
        self.outputs_dir = outputs_dir
        self.validation_dir = validation_dir
        self.verbose = verbose
        
        if self.verbose:
            print(f"Output directory: {os.path.abspath(self.outputs_dir)}")
            print(f"Validation directory: {os.path.abspath(self.validation_dir)}")
        
        # Data storage
        self.df_all = None
        self.df_val_polls = None
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'loaded_files': 0,
            'failed_files': 0,
            'models': set(),
            'levels': set(),
            'temp_configs': set()
        }
    
    def load_validation_data(self):
        """
        Load validation set data (Polling only)
        """
        polls_val_path = os.path.join(self.validation_dir, 'national_polls_results.csv')
        election_val_path = os.path.join(self.validation_dir, 'national_election_results.csv')
        
        # Try polls file first
        if os.path.exists(polls_val_path):
            try:
                self.df_val_polls = pd.read_csv(polls_val_path, sep=';', encoding='utf-8-sig')
            except:
                try:
                    self.df_val_polls = pd.read_excel(polls_val_path)
                    if self.verbose:
                        print(f"  (Read as Excel format)")
                except Exception as e:
                    if self.verbose:
                        print(f"✗ Cannot read Polls validation: {e}")
                    self.df_val_polls = None
            
            if self.df_val_polls is not None:
                self.df_val_polls['poll_percentage'] = clean_numeric(
                    self.df_val_polls['poll_percentage']
                )
                if 'candidate' in self.df_val_polls.columns:
                    self.df_val_polls['candidate'] = self.df_val_polls['candidate'].apply(
                        standardize_name
                    )
                if self.verbose:
                    print(f"✓ Loaded Polls validation: {len(self.df_val_polls)} rows")
        
        # If polls doesn't exist, try election as backup
        elif os.path.exists(election_val_path):
            try:
                self.df_val_polls = pd.read_csv(election_val_path, sep=';', encoding='utf-8-sig')
            except:
                try:
                    self.df_val_polls = pd.read_excel(election_val_path)
                    if self.verbose:
                        print(f"  (Read as Excel format)")
                except Exception as e:
                    if self.verbose:
                        print(f"✗ Cannot read Election validation: {e}")
                    self.df_val_polls = None
            
            if self.df_val_polls is not None:
                self.df_val_polls['poll_percentage'] = clean_numeric(
                    self.df_val_polls['poll_percentage']
                )
                if 'candidate' in self.df_val_polls.columns:
                    self.df_val_polls['candidate'] = self.df_val_polls['candidate'].apply(
                        standardize_name
                    )
                if self.verbose:
                    print(f"✓ Loaded Election validation (as Polling): {len(self.df_val_polls)} rows")
        else:
            if self.verbose:
                print(f"✗ Polls or Election validation not found")
    
    def scan_and_load_predictions(self) -> pd.DataFrame:
        """
        Scan outputs directory and load all prediction files (polling only)
        
        Returns: Merged complete DataFrame
        """
        # Scan files - Pattern for n{x}p{y} format
        pattern = os.path.join(self.outputs_dir, '**', '*_n[0-9]*p[0-9]*_*.csv')
        files = glob.glob(pattern, recursive=True)
        
        self.stats['total_files'] = len(files)
        
        if self.verbose:
            print(f"\nScanned {len(files)} CSV files")
        
        all_data = []
        
        for filepath in files:
            try:
                # Extract metadata
                metadata = extract_metadata_from_filename(filepath)
                
                # Skip non-polling files (only load polling)
                if metadata['task_type'] != 'polling':
                    continue
                
                # Skip files without temperature config
                if metadata['temp_config'] is None:
                    if self.verbose:
                        print(f"✗ Skipped (no temp config): {os.path.basename(filepath)}")
                    continue
                
                # Load CSV
                try:
                    df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
                except:
                    try:
                        df = pd.read_excel(filepath)
                    except Exception as read_error:
                        if self.verbose:
                            print(f"✗ Cannot read: {os.path.basename(filepath)} - {read_error}")
                        self.stats['failed_files'] += 1
                        continue
                
                # Clean numeric format
                if 'poll_percentage' in df.columns:
                    df['poll_percentage'] = clean_numeric(df['poll_percentage'])
                
                if 'confidence' in df.columns:
                    df['confidence'] = clean_numeric(df['confidence'])
                
                # Standardize candidate names
                if 'candidate' in df.columns:
                    df['candidate'] = df['candidate'].apply(standardize_name)
                
                # Add temperature column (polling uses polling_temp)
                df['temperature'] = metadata['polling_temp']
                
                # Add metadata columns
                for key, value in metadata.items():
                    df[key] = value
                
                all_data.append(df)
                
                # Update statistics
                self.stats['loaded_files'] += 1
                self.stats['models'].add(metadata['model'])
                self.stats['levels'].add(metadata['level'])
                self.stats['temp_configs'].add(metadata['temp_config'])
                
            except Exception as e:
                self.stats['failed_files'] += 1
                if self.verbose:
                    print(f"✗ Load failed: {os.path.basename(filepath)} - {e}")
        
        # Merge all data
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            
            if self.verbose:
                print(f"\n✓ Successfully loaded {self.stats['loaded_files']} files")
                print(f"  Models: {sorted(self.stats['models'])}")
                print(f"  Levels: {sorted(self.stats['levels'])}")
                print(f"  Temps: {sorted(self.stats['temp_configs'])}")
                print(f"\nTotal {len(df_all)} prediction records")
            
            return df_all
        else:
            if self.verbose:
                print("✗ No data loaded!")
            return pd.DataFrame()
    
    def join_with_validation(self, df_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Join prediction data with validation data
        
        Join rules for Polling: ['month', 'party', 'candidate']
        
        Added columns:
        - actual: Ground truth value
        - error: predicted - actual
        - abs_error: |error|
        - squared_error: error^2
        """
        if df_pred.empty or self.df_val_polls is None:
            return pd.DataFrame()
        
        df_merged = df_pred.merge(
            self.df_val_polls[['month', 'party', 'candidate', 'poll_percentage']],
            on=['month', 'party', 'candidate'],
            how='left',
            suffixes=('', '_actual')
        )
        df_merged.rename(columns={'poll_percentage_actual': 'actual'}, inplace=True)
        
        if self.verbose:
            matched = df_merged['actual'].notna().sum()
            total = len(df_merged)
            print(f"\nPolling matched: {matched}/{total} ({matched/total*100:.1f}%)")
        
        # Calculate error columns
        df_merged['error'] = df_merged['poll_percentage'] - df_merged['actual']
        df_merged['abs_error'] = df_merged['error'].abs()
        df_merged['squared_error'] = df_merged['error'] ** 2
        
        return df_merged
    
    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate evaluation metrics
        
        Grouping: [model, level, temp_config]
    
        """
        # Keep only data with ground truth
        df_valid = df[df['actual'].notna()].copy()
        
        if df_valid.empty:
            if self.verbose:
                print("\n✗ No matched validation data, cannot calculate metrics")
            return pd.DataFrame()
        
        # Grouping (no version, no task_type since polling only)
        group_keys = ['model', 'level', 'temp_config']
        
        metrics = []
        
        for name, group in df_valid.groupby(group_keys):
            model, level, temp_config = name
            
            # Basic metrics
            mae = group['abs_error'].mean()
            mse = group['squared_error'].mean()
            rmse = np.sqrt(mse)
            bias = group['error'].mean()
            count = len(group)
            
            # Pearson correlation
            try:
                if len(group) > 1:
                    r, p_value = pearsonr(group['poll_percentage'], group['actual'])
                else:
                    r, p_value = np.nan, np.nan
            except:
                r, p_value = np.nan, np.nan
            
            # Average confidence (if exists)
            if 'confidence' in group.columns and group['confidence'].notna().any():
                avg_confidence = group['confidence'].mean()
            else:
                avg_confidence = np.nan
            
            metrics.append({
                'model': model,
                'level': level,
                'temp_config': temp_config,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'Pearson_r': r,
                'Pearson_p': p_value,
                'Bias': bias,
                'Avg_Confidence': avg_confidence,
                'Count': count
            })
        
        df_metrics = pd.DataFrame(metrics)
        
        if self.verbose:
            print(f"\n✓ Calculated metrics for {len(df_metrics)} configurations")
        
        return df_metrics
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute complete loading workflow
        
        Returns:
        - df_all: Complete data (predictions, ground truth, errors)
        - df_metrics: Aggregated metrics
        """
        if self.verbose:
            print("=" * 60)
            print("Temperature Scan Experiment Data Loading (Polling Only)")
            print("=" * 60)
        
        # 1. Load validation sets
        self.load_validation_data()
        
        # 2. Load prediction data
        df_pred = self.scan_and_load_predictions()
        
        if df_pred.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 3. Join validation data
        df_all = self.join_with_validation(df_pred)
        
        # 4. Calculate metrics
        df_metrics = self.calculate_metrics(df_all)
        
        # Save
        self.df_all = df_all
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Data loading complete!")
            print("=" * 60)
        
        return df_all, df_metrics
    
    def filter_data(self, models: Optional[List[str]] = None,
                    levels: Optional[List[str]] = None,
                    temp_configs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter data
        
        Returns: Filtered DataFrame
        """
        if self.df_all is None:
            raise ValueError("Please call load_all() first to load data")
        
        df = self.df_all.copy()
        
        if models:
            df = df[df['model'].isin(models)]
        if levels:
            df = df[df['level'].isin(levels)]
        if temp_configs:
            df = df[df['temp_config'].isin(temp_configs)]
        
        return df


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_load(outputs_dir: str = None,
               validation_dir: str = None,
               verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - df_all: Complete data
    - df_metrics: Aggregated metrics
    """
    loader = TemperatureExperimentLoader(
        outputs_dir=outputs_dir,
        validation_dir=validation_dir,
        verbose=verbose
    )
    return loader.load_all()


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    # Test loading
    df_all, df_metrics = quick_load()
    
    print("\n" + "=" * 60)
    print("Data Preview")
    print("=" * 60)
    
    if not df_all.empty:
        print("\nComplete data (first 5 rows):")
        print(df_all.head())
        
        print("\nColumns:")
        print(df_all.columns.tolist())
    
    if not df_metrics.empty:
        print("\nMetrics summary (first 10 rows):")
        print(df_metrics.head(10))
        
        print("\nBest configurations (sorted by MAE):")
        print(df_metrics.nsmallest(5, 'MAE')[['model', 'level', 'temp_config', 
                                                'MAE', 'Pearson_r']])