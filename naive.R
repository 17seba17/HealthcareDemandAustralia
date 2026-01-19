
library(pscl) 

df <- read.csv('healthcare_cleaned.csv')

all_features <- c(
  'sex', 'age', 'income', 'levyplus', 'freepor', 'freerepa',
  'illness', 'actdays', 'hscore', 'chcond1', 'chcond2', 
  'nondocco', 'hospadmi', 'hospdays', 'prescrib', 'nonprescr', 
  'any_insurance', 'health_status_severe', 'recent_hospital', 
  'medication_user', 'any_chronic'
)

formula_str <- paste("dvisits ~", paste(all_features, collapse = " + "))
f <- as.formula(formula_str)

print(paste("Formula used:", formula_str))


model_poisson <- glm(f, data = df, family = poisson(link = "log"))

summary_model <- summary(model_poisson)
print(summary_model)


results <- as.data.frame(summary_model$coefficients)

colnames(results) <- c("Estimate", "StdError", "Z_Value", "P_Value")

results$Variable <- rownames(results)

cat("_____")

  library(MASS)
library(lmtest)


model_poisson <- glm(f, data = df, family = poisson(link = "log"))

model_nb <- glm.nb(f, data = df)

summary(model_nb)
cat("_____")

test <- lrtest(model_poisson, model_nb)

print(test)

cat("_____")


print(summary(model_nb))

print(coef(summary(model_nb)))