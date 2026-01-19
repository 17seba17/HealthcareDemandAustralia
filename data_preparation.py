"""
Healthcare Demand in Australia - Data Preparation and Treatment
================================================================

Project: Predicting Doctor Visits Using Regression Models
Dataset: Australian Health Survey 1977-1978
Target Variable: dvisits (Number of doctor consultations in past 2 weeks)

This script handles:
- Data loading from R data file (.rda)
- Exploratory Data Analysis (EDA)
- Data cleaning and quality checks
- Feature engineering and transformation
- Preparation for regression modeling (Poisson, Negative Binomial, GLM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)


class HealthcareDataPreparation:
    """
    Main class for healthcare data preparation and treatment.
    
    Attributes:
        data_path (str): Path to the R data file
        df (pd.DataFrame): Main dataframe containing the data
        df_clean (pd.DataFrame): Cleaned dataframe ready for modeling
        metadata (dict): Dictionary storing data characteristics and statistics
    """
    
    def __init__(self, data_path):
        """
        Initialize the data preparation pipeline.
        
        Parameters:
            data_path (str): Path to the .rda file
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.metadata = {
            'original_shape': None,
            'clean_shape': None,
            'missing_values': {},
            'zero_inflation': {},
            'outliers': {},
            'transformations': []
        }
        
    def load_data(self):
        """
        Load data from R data file (.rda) format.
        
        Uses pyreadr library to read .rda files.
        Falls back to rpy2 if pyreadr is not available.
        
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("STEP 1: LOADING DATA")
        try:
            import pyreadr
            result = pyreadr.read_r(self.data_path)
            # Get the dataframe from the result
            self.df = result[list(result.keys())[0]]
            
        except ImportError:
            try:
                from rpy2.robjects import r, pandas2ri
                pandas2ri.activate()
                r['load'](self.data_path)
                # Get the loaded object name
                obj_name = r('ls()')[0]
                self.df = pandas2ri.rpy2py(r[obj_name])
            except ImportError:
                raise ImportError(
                    "Neither pyreadr nor rpy2 is installed. "
                )
        
        # Rename columns to match demandhealthcareaustralia.txt documentation
        column_mapping = {
            'doctorco': 'dvisits',
            'freepoor': 'freepor',
            'freepera': 'freerepa',
            'nonpresc': 'nonprescr'
        }
        self.df = self.df.rename(columns=column_mapping)      
        self.metadata['original_shape'] = self.df.shape
        print(f"Dataset shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nColumn names:\n{self.df.columns.tolist()}")
        print("\n")
        
        return self.df
    
    def initial_exploration(self):
        """
        Perform initial exploratory data analysis.
        
        Displays:
        - First few rows
        - Data types
        - Basic statistics
        - Missing value counts
        """
        print("STEP 2: INITIAL DATA EXPLORATION")
        
        print("\n--- First 5 rows ---")
        print(self.df.head())
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- Basic Statistics ---")
        print(self.df.describe())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing_df) > 0:
            print(missing_df)
            self.metadata['missing_values'] = missing_df.to_dict()
        else:
            print("No missing values found.")
            
        print("\n")
        
    def analyze_target_variable(self):
        """
        Analyze the target variable (dvisits) in detail.
        
        Examines:
        - Distribution
        - Zero-inflation
        - Overdispersion
        - Summary statistics
        """
        print("STEP 3: TARGET VARIABLE ANALYSIS (dvisits)")

        target = 'dvisits'
        
        print(f"\n--- Summary Statistics for {target} ---")
        print(self.df[target].describe())
        
        # Zero-inflation analysis
        zero_count = (self.df[target] == 0).sum()
        zero_pct = (zero_count / len(self.df)) * 100
        print(f"\nZero values: {zero_count} ({zero_pct:.2f}%)")
        
        # Value distribution
        print(f"\nValue counts (first 20):")
        print(self.df[target].value_counts().sort_index().head(20))
        
        # Overdispersion check
        mean_val = self.df[target].mean()
        var_val = self.df[target].var()
        print(f"\nMean: {mean_val:.3f}")
        print(f"Variance: {var_val:.3f}")
        print(f"Variance/Mean ratio: {var_val/mean_val:.3f}")
        
        if var_val > mean_val:
            print("-> OVERDISPERSION detected (Variance > Mean)")
            print("Consider Negative Binomial or Zero-Inflated models")
        else:
            print("-> Equidispersion or underdispersion")
            print("Poisson model may be appropriate")
        
        self.metadata['zero_inflation'][target] = {
            'zero_count': int(zero_count),
            'zero_percentage': float(zero_pct),
            'mean': float(mean_val),
            'variance': float(var_val),
            'dispersion_ratio': float(var_val/mean_val)
        }
        
        print("\n")
        
    def analyze_count_variables(self):
        """
        Analyze all count variables in the dataset.
        
        Count variables: illness, actdays, nondocco, hospadmi, 
                        medicine, prescrib, nonprescr
        """
        print("STEP 4: COUNT VARIABLES ANALYSIS")
        
        count_vars = ['illness', 'actdays', 'nondocco', 'hospadmi', 
                     'medicine', 'prescrib', 'nonprescr']
        
        results = []
        for var in count_vars:
            if var in self.df.columns:
                zero_count = (self.df[var] == 0).sum()
                zero_pct = (zero_count / len(self.df)) * 100
                mean_val = self.df[var].mean()
                var_val = self.df[var].var()
                max_val = self.df[var].max()
                
                results.append({
                    'Variable': var,
                    'Mean': mean_val,
                    'Variance': var_val,
                    'Var/Mean': var_val/mean_val if mean_val > 0 else 0,
                    'Max': max_val,
                    'Zeros': zero_count,
                    'Zero%': zero_pct
                })
                
                self.metadata['zero_inflation'][var] = {
                    'zero_count': int(zero_count),
                    'zero_percentage': float(zero_pct)
                }
        
        results_df = pd.DataFrame(results)
        print("\n", results_df.to_string(index=False))
        print("\n")
        
    def analyze_binary_variables(self):
        """
        Analyze binary/indicator variables.
        
        Binary variables: sex, levyplus, freepor, freerepa, chcond1, chcond2
        """
        print("STEP 5: BINARY VARIABLES ANALYSIS")
        
        binary_vars = ['sex', 'levyplus', 'freepor', 'freerepa', 'chcond1', 'chcond2']
        
        for var in binary_vars:
            if var in self.df.columns:
                print(f"\n--- {var} ---")
                print(self.df[var].value_counts())
                print(f"Proportion of 1s: {(self.df[var] == 1).sum() / len(self.df) * 100:.2f}%")
        
        print("\n")
        
    def analyze_continuous_variables(self):
        """
        Analyze continuous/scaled variables.
        
        Continuous variables: age, agesq, income, hscore, hospdays
        """
        print("STEP 6: CONTINUOUS VARIABLES ANALYSIS")
        
        continuous_vars = ['age', 'agesq', 'income', 'hscore', 'hospdays']
        
        for var in continuous_vars:
            if var in self.df.columns:
                print(f"\n--- {var} ---")
                print(self.df[var].describe())
                
                # Check for potential outliers using IQR method
                Q1 = self.df[var].quantile(0.25)
                Q3 = self.df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[var] < lower_bound) | (self.df[var] > upper_bound)]
                outlier_pct = len(outliers) / len(self.df) * 100
                
                print(f"Potential outliers (IQR method): {len(outliers)} ({outlier_pct:.2f}%)")
                
                self.metadata['outliers'][var] = {
                    'count': int(len(outliers)),
                    'percentage': float(outlier_pct)
                }
        
        print("\n")
        
    def check_multicollinearity(self):
        """
        Check for multicollinearity issues among predictors.
        
        Examines:
        - Correlation matrix
        - Perfect collinearity (medicine = prescrib + nonprescr)
        - High correlations (>0.8)
        """

        print("STEP 7: MULTICOLLINEARITY ANALYSIS")

        
        # Check perfect collinearity: medicine = prescrib + nonprescr
        print("\n--- Checking Perfect Collinearity ---")
        if all(col in self.df.columns for col in ['medicine', 'prescrib', 'nonprescr']):
            sum_check = self.df['prescrib'] + self.df['nonprescr']
            is_perfect = np.allclose(self.df['medicine'], sum_check)
            
            if is_perfect:
                print("CONFIRMED: medicine = prescrib + nonprescr (perfect collinearity)")
                print("So, we remove 'medicine' variable to avoid redundancy")
            else:
                print("No perfect collinearity detected between medicine and prescrib+nonprescr")
        
        # Correlation matrix for all numeric variables
        print("\n--- Correlation Matrix (High Correlations is > 0.8) ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append({
                        'Var1': corr_matrix.columns[i],
                        'Var2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr)
            print(high_corr_df.to_string(index=False))
        else:
            print("No high correlations found between variables")
        
        # Age and agesq correlation
        if 'age' in self.df.columns and 'agesq' in self.df.columns:
            age_corr = self.df[['age', 'agesq']].corr().iloc[0, 1]
            print(f"\nCorrelation between age and agesq: {age_corr:.3f}")        
        print("\n")
        
    def clean_data(self):
        """
        Clean and prepare data for modeling.
        
        Steps:
        - Remove perfect collinearity (medicine variable)
        - Handle missing values if any
        - Create clean dataset
        """

        print("STEP 8: DATA CLEANING")

        
        # Create a copy for cleaning
        self.df_clean = self.df.copy()
        
        # Remove 'medicine' and 'constant' variables
        cols_to_remove = []
        if 'medicine' in self.df_clean.columns:
            print("\nRemoving 'medicine' variable (redundant: medicine = prescrib + nonprescr)")
            cols_to_remove.append('medicine')
            self.metadata['transformations'].append('Removed medicine variable (perfect collinearity)')
        if 'constant' in self.df_clean.columns:
            print("Removing 'constant' variable (all values are 1)")
            cols_to_remove.append('constant')
            self.metadata['transformations'].append('Removed constant variable (no information)')
        if cols_to_remove:
            self.df_clean = self.df_clean.drop(cols_to_remove, axis=1)
        
        # Handle missing values if present
        missing_count = self.df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"\nHandling {missing_count} missing values...")
            # Strategy: Remove rows with missing values if < 5% of data
            missing_pct = (missing_count / (self.df_clean.shape[0] * self.df_clean.shape[1])) * 100
            
            if missing_pct < 5:
                self.df_clean = self.df_clean.dropna()
                print(f"Removed rows with missing values ({missing_pct:.2f}% of total data)")
                self.metadata['transformations'].append(f'Removed rows with missing values ({missing_pct:.2f}%)')
        else:
            print("\nNo missing values to handle.")
        
        self.metadata['clean_shape'] = self.df_clean.shape
        print(f"\nCleaned dataset shape: {self.df_clean.shape[0]} rows, {self.df_clean.shape[1]} columns")
        print("\n")
        
    def create_additional_features(self):
        """
        Create additional features for potential model improvements.
        
        New features:
        - any_insurance: Indicator if covered by any insurance (levyplus, freepor, or freerepa)
        - health_status_severe: Indicator for severe health issues (chcond2=1 or hscore high)
        - recent_hospital: Indicator if hospitalized recently (hospadmi > 0)
        - medication_user: Indicator if using any medication (prescrib + nonprescr > 0)
        """

        print("STEP 9: FEATURE ENGINEERING")

        
        print("\nCreating additional features...")
        
        # Any insurance coverage
        if all(col in self.df_clean.columns for col in ['levyplus', 'freepor', 'freerepa']):
            self.df_clean['any_insurance'] = (
                (self.df_clean['levyplus'] == 1) | 
                (self.df_clean['freepor'] == 1) | 
                (self.df_clean['freerepa'] == 1)
            ).astype(int)
            print("- Created 'any_insurance': 1 if covered by any insurance type")
            self.metadata['transformations'].append('Created any_insurance feature')
        
        # Severe health status (chcond2=1 indicates chronic condition with activity limitation)
        if 'chcond2' in self.df_clean.columns:
            self.df_clean['health_status_severe'] = self.df_clean['chcond2']
            print("- Created 'health_status_severe': Same as chcond2 (chronic condition with limitation)")
            self.metadata['transformations'].append('Created health_status_severe feature')
        # Recent hospitalization
        if 'hospadmi' in self.df_clean.columns:
            self.df_clean['recent_hospital'] = (self.df_clean['hospadmi'] > 0).astype(int)
            print("- Created 'recent_hospital': 1 if hospitalized in past 12 months")
            self.metadata['transformations'].append('Created recent_hospital feature')
        
        # Medication user
        if all(col in self.df_clean.columns for col in ['prescrib', 'nonprescr']):
            self.df_clean['medication_user'] = (
                (self.df_clean['prescrib'] + self.df_clean['nonprescr']) > 0
            ).astype(int)
            print("- Created 'medication_user': 1 if using any medication")
            self.metadata['transformations'].append('Created medication_user feature')
        
        # Total chronic conditions
        if all(col in self.df_clean.columns for col in ['chcond1', 'chcond2']):
            self.df_clean['any_chronic'] = (
                (self.df_clean['chcond1'] == 1) | (self.df_clean['chcond2'] == 1)
            ).astype(int)
            print("- Created 'any_chronic': 1 if has any chronic condition")
            self.metadata['transformations'].append('Created any_chronic feature')
        
        print(f"\nFinal dataset shape: {self.df_clean.shape[0]} rows, {self.df_clean.shape[1]} columns")
        print("\n")
        
    def generate_summary_report(self):
        """
        Generate comprehensive summary report of the cleaned dataset.
        """
        print("STEP 10: FINAL SUMMARY REPORT")
      
        print(f"\n--- Dataset Overview ---")
        print(f"Original shape: {self.metadata['original_shape']}")
        print(f"Final shape: {self.metadata['clean_shape']}")
        print(f"Rows removed: {self.metadata['original_shape'][0] - self.metadata['clean_shape'][0]}")
        print(f"Columns removed: {self.metadata['original_shape'][1] - self.metadata['clean_shape'][1]}")
        
        print(f"\n--- Transformations Applied ---")
        for i, transform in enumerate(self.metadata['transformations'], 1):
            print(f"{i}. {transform}")
        
        print(f"\n--- Target Variable (dvisits) Characteristics ---")
        if 'dvisits' in self.metadata['zero_inflation']:
            target_info = self.metadata['zero_inflation']['dvisits']
            print(f"Mean: {target_info['mean']:.3f}")
            print(f"Variance: {target_info['variance']:.3f}")
            print(f"Dispersion ratio: {target_info['dispersion_ratio']:.3f}")
            print(f"Zero-inflation: {target_info['zero_percentage']:.2f}%")
        
        print(f"\n--- Recommended Models ---")
        if 'dvisits' in self.metadata['zero_inflation']:
            dispersion = self.metadata['zero_inflation']['dvisits']['dispersion_ratio']
            zero_pct = self.metadata['zero_inflation']['dvisits']['zero_percentage']
            
            print("Based on data characteristics:")
            if dispersion > 2 and zero_pct > 30:
                print("1. Zero-Inflated Negative Binomial (ZINB) - HIGHLY RECOMMENDED")
                print("2. Negative Binomial Regression")
                print("3. Zero-Inflated Poisson (ZIP)")
            elif dispersion > 2:
                print("1. Negative Binomial Regression - RECOMMENDED")
                print("2. Poisson Regression (for comparison)")
            elif zero_pct > 30:
                print("1. Zero-Inflated Poisson (ZIP) - RECOMMENDED")
                print("2. Poisson Regression")
            else:
                print("1. Poisson Regression - RECOMMENDED")
                print("2. Negative Binomial Regression (for robustness)")
        
        print(f"\n--- Variable Categories ---")
        print(f"\nTarget variable:")
        print("  - dvisits (number of doctor consultations)")
        
        print(f"\nDemographic variables:")
        demographic = ['sex', 'age', 'agesq', 'income']
        print(f"  - {', '.join([v for v in demographic if v in self.df_clean.columns])}")
        
        print(f"\nInsurance variables:")
        insurance = ['levyplus', 'freepor', 'freerepa', 'any_insurance']
        print(f"  - {', '.join([v for v in insurance if v in self.df_clean.columns])}")
        
        print(f"\nHealth status variables:")
        health = ['illness', 'actdays', 'hscore', 'chcond1', 'chcond2', 'any_chronic', 'health_status_severe']
        print(f"  - {', '.join([v for v in health if v in self.df_clean.columns])}")
        
        print(f"\nHealthcare utilization variables:")
        utilization = ['nondocco', 'hospadmi', 'hospdays', 'recent_hospital']
        print(f"  - {', '.join([v for v in utilization if v in self.df_clean.columns])}")
        
        print(f"\nMedication variables:")
        medication = ['prescrib', 'nonprescr', 'medication_user']
        print(f"  - {', '.join([v for v in medication if v in self.df_clean.columns])}")
        
        print("\n")
        
    def save_cleaned_data(self, output_path):
        """
        Save cleaned dataset to CSV file.
        
        Parameters:
            output_path (str): Path to save the CSV file
        """
        print("STEP 11: SAVING CLEANED DATA")
        
        self.df_clean.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        print(f"Shape: {self.df_clean.shape}")
        print("\n")
        
    def run_full_pipeline(self, output_csv_path=None):
        """
        Run the complete data preparation pipeline.
        
        Parameters:
            output_csv_path (str): Path to save cleaned data (optional)
        """
        print("\n" + "="*70)
        print("HEALTHCARE DATA PREPARATION PIPELINE")
        print("Australian Health Survey 1977-1978")
        print("="*70 + "\n")
        
        # Execute all steps
        self.load_data()
        self.initial_exploration()
        self.analyze_target_variable()
        self.analyze_count_variables()
        self.analyze_binary_variables()
        self.analyze_continuous_variables()
        self.check_multicollinearity()
        self.clean_data()
        self.create_additional_features()
        self.generate_summary_report()
        
        if output_csv_path:
            self.save_cleaned_data(output_csv_path)
        
        print("\nNext steps:")
        print("1. Explore data visualizations (distributions, correlations)")
        print("2. Split data into training and testing sets")
        print("3. Implement regression models (Poisson, Negative Binomial, GLM)")
        print("4. Compare model performance using AIC, BIC, and cross-validation because its important for results discussion")
        print("5. Interpret coefficients and generate predictions")
        print("\n")
        
        return self.df_clean


def main():
    """
    Main execution function.
    """
    # Define paths relative to script location (universal for any user)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "HealthCareAustralia", "HealthCareAustralia.rda")
    output_path = os.path.join(script_dir, "healthcare_cleaned.csv")
    
    # Initialize and run pipeline
    pipeline = HealthcareDataPreparation(data_path)
    df_clean = pipeline.run_full_pipeline(output_csv_path=output_path)
    
    # Return cleaned dataframe for further use
    return df_clean, pipeline


if __name__ == "__main__":
    df_clean, pipeline = main()

