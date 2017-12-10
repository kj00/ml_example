##
source("script/input_to_intermidiate/prepare.R")

##
library(xgboost)

##
bst <- xgboost(
  data = as.matrix(x_train_tbl),
  label = y_train_vec,
  max_depth = 8,
  eta = 1.2,
  lambda = 100,
  nthread = 2,
  nrounds = 80,
  colsample_bytree = 1,
  colsample_bylevel = 1,
  objective = "binary:logistic"
)


#
pred <- predict(bst, as.matrix(x_test_tbl))
#hist(pred)

##
# Format test data and predictions for yardstick metrics
library(forcats)
estimates_xgb_tbl <- tibble(
  truth      =  as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.integer(pred >= 0.5) %>%
    as.factor() %>%
    fct_recode(yes = "1", no = "0"),
  class_prob = pred
) %>% 
  arrange(-class_prob) %>% 
  mutate(test = 
           (purrr::accumulate(estimate == "yes" & truth == "yes", `+`) +
              purrr::accumulate(estimate == "no" & truth == "yes", `+`))/ sum(truth == "yes"))

# Confusion Table
estimates_xgb_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_xgb_tbl %>% metrics(truth, estimate)
# AUC
estimates_xgb_tbl %>% roc_auc(truth, class_prob)

#loglos
estimates_l2_tbl %>%
  mutate(class_prob_no = 1-class_prob) %>% 
  mnLogLoss(truth, class_prob, class_prob_no)

# Precision
tibble(
  precision = estimates_xgb_tbl %>% precision(truth, estimate),
  recall    = estimates_xgb_tbl %>% recall(truth, estimate)
)


##
# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl, 
  model          = bstSparse, 
  bin_continuous = FALSE)


##
importance_matrix <- xgb.importance(model = bstSparse)
print(importance_matrix)


for (i in seq_along(importance_matrix$Feature)){
  importance_matrix$Feature[i] <- colnames(x_train_tbl)[i]
  }

xgb.plot.importance(importance_matrix = importance_matrix)

