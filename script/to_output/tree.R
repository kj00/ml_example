source("script/input_to_intermidiate/prepare.R")

library(rpart)


dt_1 <- rpart(Churn ~ .,
              train_tbl)

pred <- predict(dt_1, test_tbl)[, "Yes"]
hist(pred)

# Format test data and predictions for yardstick metrics
library(forcats)

estimates_tree_tbl <- tibble(
  truth      =  as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.integer(pred >= 0.5) %>%
    as.factor() %>%
    fct_recode(yes = "1", no = "0"),
  class_prob = pred
)

estimates_tree_tbl

# Confusion Table
estimates_tree_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_tree_tbl %>% metrics(truth, estimate)
# AUC
estimates_tree_tbl %>% roc_auc(truth, class_prob)

# Precision
tibble(
  precision = estimates_tree_tbl %>% precision(truth, estimate),
  recall    = estimates_tree_tbl %>% recall(truth, estimate)
)
