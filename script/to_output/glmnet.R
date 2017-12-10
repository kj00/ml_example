library(glmnet)

l2lg <- cv.glmnet(x = x_train_tbl %>% as.matrix(),
       y = y_train_vec %>% as.factor,
       family = "binomial")

plot(l2lg)
pred <- predict(l2lg, x_test_tbl %>% as.matrix,
          s='lambda.min',
          type="response") 

# Format test data and predictions for yardstick metrics
library(forcats)
estimates_l2_tbl <- tibble(
  truth      =  as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.integer(pred >= 0.5) %>%
    as.factor() %>%
    fct_recode(yes = "1", no = "0"),
  class_prob = pred[, 1]
) %>% 
  arrange(-class_prob) %>% 
  mutate(test = 
           (purrr::accumulate(estimate == "yes" & truth == "yes", `+`) +
              purrr::accumulate(estimate == "no" & truth == "yes", `+`))/ sum(truth == "yes"))


# Confusion Table
estimates_l2_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_l2_tbl %>% metrics(truth, estimate)
# AUC
estimates_l2_tbl %>% roc_auc(truth, class_prob)

#loglos
estimates_l2_tbl %>%
  mutate(class_prob_no = 1-class_prob) %>% 
  mnLogLoss(truth, class_prob, class_prob_no)


# Precision
tibble(
  precision = estimates_xgb_tbl %>% precision(truth, estimate),
  recall    = estimates_xgb_tbl %>% recall(truth, estimate)
)


