#
source("script/input_to_intermidiate/prepare.R")

#
lg_1 <- glm(Churn ~ .,
    data = train_tbl %>% 
      mutate(Churn = ifelse(Churn == "Yes", 1, 0)),
    family = binomial(link = "logit")) %>% 
  stats::step(direction = "backward")

##
pred <- predict(lg_1,
                model.frame(formula(lg_1),
                            test_tbl), type = "response")
hist(pred)

# Format test data and predictions for yardstick metrics
library(forcats)

estimates_lg_tbl <- tibble(
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
estimates_lg_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_lg_tbl %>% metrics(truth, estimate)
# AUC
estimates_lg_tbl %>% roc_auc(truth, class_prob)

# Precision
tibble(
  precision = estimates_lg_tbl %>% precision(truth, estimate),
  recall    = estimates_lg_tbl %>% recall(truth, estimate)
)

