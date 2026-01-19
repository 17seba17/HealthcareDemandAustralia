import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd



df = pd.read_csv('healthcare_cleaned.csv')
all_features = [
    'sex', 'age', 'income', 'levyplus'
    , 'freepor', 'freerepa','illness', 'actdays',
     'hscore', 'chcond1', 'chcond2', 'nondocco', 
    'hospadmi', 'hospdays', 'prescrib', 'nonprescr', 
    'any_insurance', 'health_status_severe', 'recent_hospital', 'medication_user', 
    'any_chronic'
]
formula_string = 'dvisits ~ ' + ' + '.join(all_features)

print(f"Formula used: {formula_string}")

model = smf.glm(formula=formula_string, data=df, family=sm.families.Poisson()).fit()

print(model.summary())