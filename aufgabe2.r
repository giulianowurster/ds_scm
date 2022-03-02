###Setup
rm(list = ls())
set.seed(616)


##Install and load all required packages.
list_of_packages <- c("data.table", "naniar", "mice", "caret", "DescTools", "magrittr", "foreach", "doParallel", "parallel", "purrr", "tibble")
new_packages <- list_of_packages[!(list_of_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
lapply(list_of_packages, require, character.only = TRUE)
rm(list_of_packages, new_packages)



###Register cores for parralel processing.\
cl <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel(cl)



###Load data.
main_data <- data.table::fread("data_recycling.csv") %>% tibble::as_tibble()



###Remove outliers.
##Visual detection of outliers.
for (i in 1:ncol(main_data[,-4])) {
  hist(main_data[,i], labels = names(main_data[,i]))
}
rm(i)

##Manually removing outlier(s).
main_data <- main_data[-50,]



###Remove empty rows (and columns).
##rows
main_data <- main_data[!apply(is.na(main_data) | main_data == "", 1, all),]


##columns
#main_data <- main_data[,!apply(is.na(main_data) | main_data == "", 2, all), with = FALSE]



###Impute missings.
##Test for MCAR.

#Disabled for safety reasons!
#Also not needed for this data.
# doi: 10.1080/01621459.1988.10478722
# doi: 10.18637/jss.v056.i06
#library("devtools")
#install_github("cran/MissMech")
#library("MissMech")

naniar::mcar_test(main_data)["p.value"] #MCAR if > 0.05
#MissMech::TestMCARNormality(main_data)[["pvalcomb"]]


##Imputation with MICE, since there is evidence, that our data is MCAR.
quantity_of_imputations = 5
imp <- mice::mice(as.matrix(main_data),
                  method = "norm",
                  m = quantity_of_imputations,
                  print = FALSE)

main_data_imputed_model <- mice::mice(main_data,
                                      method = "norm",
                                      predictorMatrix = imp$predictorMatrix,
                                      m = quantity_of_imputations)

main_data_imputed_all <- mice::complete(main_data_imputed_model, action = "all")
rm(imp)


###Model training with cross-validation (k =10).
##Simple way to determine the parameters for lm().
all_variable_combinations <- DescTools::CombSet(names(main_data[,-4]), 1:3, as.list = TRUE) #get all var. comb.

# all_correlations <- foreach::foreach(j = all_variable_combinations) %do% {
#   data_to_be_operated_on <- foreach::foreach(k = j) %do% main_data[[k]] #get data for all  var. comb.
#   data_to_be_operated_on %<>% unlist() %>% matrix(nrow = nrow(main_data)) #turn the data back into a matrix
#   product_vector <- apply(data_to_be_operated_on, 1, prod) #get the product of the vectors
#   correlation <- cor(main_data$Rate, product_vector, use = "na.or.complete") #correlations of those products and the Rate
#   return(correlation)
# }
#
# all_variable_combinations_with_correlation <- Map(list,
#                                                    correlation = all_correlations,
#                                                    parameters = all_variable_combinations)

rm(all_correlations, j, k, data_to_be_operated_on, product_vector, correlation)


##... or just loop over all possibilities.
#Get the different formulas for every model.
all_variable_combinations_as_formula <-
  all_variable_combinations %>% purrr::map(function(x) {
    x %>%
      unlist() %>%
      paste(collapse = " * ") %>%
      paste(names(main_data[, 4]), ., sep = " ~ ")})

##Do the acctual training.
ctrl <- caret::trainControl(method = "repeatedcv", repeats = 3)

foreach::foreach(current_formula = all_variable_combinations_as_formula) %dopar% {
  
  library(magrittr, include.only = "%>%") #Required, since we do parallel processing.
  
all_models <- main_data_imputed_all %>% purrr::map( #training models on each imputed data set
  ~ caret::train(data = .,
                 as.formula(current_formula),
                 method = "lm",
                 tuneLength = 10,
                 trControl = ctrl))

pooled_coefficients <- all_models %>% purrr::map(function(x) { #Pool the estimates from each model.
  x %>%
    summary() %>%
    .[["coefficients"]] %>%
    .[1:nrow(.)]}) %>%
  unlist() %>%
  matrix(ncol = quantity_of_imputations) %>%
  apply(1, mean)

pooled_R2 <- all_models %>% purrr::map(function(x) {  #pool R2 with tranformation; 
  x$results$Rsquared %>%                              #doi: 10.1080/02664760802553000
    sqrt() %>%
    DescTools::FisherZ()}) %>%
  unlist() %>%
  mean() %>%
  DescTools::FisherZInv() %>%
  '^'(2)

return(list(
  formula = current_formula,
  coefficients = pooled_coefficients,
  R2 = pooled_R2))
} -> results


best_result <- results %>% purrr::map(3) %>% which.max() #Find the model with highest R2.

cat("Das beste Ergebnis zum Vorhersagen der Rate liefter die lineare Regression mit der Formel: ",
    as.character(results[[best_result]][1]),".\n",
    "Die Betas lauten: ",
    as.character(results[[best_result]][2]),".\n",
    "Mit einem R2 von: ",
    as.character(results[[best_result]][3]),".",
    sep = "")

stopCluster(cl) #Close cluster.
