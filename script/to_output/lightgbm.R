##
source("script/input_to_intermidiate/prepare.R")

##
library(lightgbm)

##
lgb <- lgb.train(
   data = lgb.Dataset(data.matrix(x_train_tbl),
                     label = y_train_vec),
  valids = list(validation = lgb.Dataset(data.matrix(x_test_tbl),
                                 label = y_test_vec)),
  nrounds = 30000,
  obj = "binary",
  params = list(
    num_leaves = 10,
    min_data_in_leaf = 50,
    num_threads = 2,
    boosting = "dart",
    learning_rate = 0.005,
    bagging_fraction = 0.8,
    feature_fraction = 0.7,
    early_stopping_round = 2500,
    tree_learner = "feature",
    metric = "auc",
    metric_freq = 0.1)
)

lgb.save(lgb, "data/output/model/lgb_binary_1")


##
pred <- predict(lgb, data.matrix(x_test_tbl))

#hist(pred)

##
# Format test data and predictions for yardstick metrics
library(forcats)
estimates_xgb_tbl <- tibble(
  truth      =  factor(y_test_vec, levels = c(1, 0)) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.integer(pred >= 0.5) %>%
    factor(levels = c(1, 0)) %>%
    fct_recode(yes = "1", no = "0"),
  class_prob = pred
) %>%
  arrange(-class_prob) %>%
  mutate(test =
           (
             purrr::accumulate(estimate == "yes" & truth == "yes", `+`) +
               purrr::accumulate(estimate == "no" &
                                   truth == "yes", `+`)
           ) / sum(truth == "yes"))

# Confusion Table
estimates_xgb_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_xgb_tbl %>% metrics(truth, estimate)
# AUC
estimates_xgb_tbl %>% roc_auc(truth, class_prob)

#loglos
estimates_xgb_tbl %>%
  mutate(class_prob_no = 1 - class_prob) %>%
  mnLogLoss(truth, class_prob, class_prob_no)

# Precision
tibble(
  precision = estimates_xgb_tbl %>% precision(truth, estimate),
  recall    = estimates_xgb_tbl %>% recall(truth, estimate)
)


##
plot(
  y = estimates_xgb_tbl$test,
  x = seq_len(length(estimates_xgb_tbl$test)) / length(estimates_xgb_tbl$test),
  type = "l"
)
lines(x = seq(0, 1, 0.1),
      y = seq(0, 1, 0.1))

##
library(lime)
class(lgb)

# Setup lime::model_type() function for keras
model_type.lgb.Booster <- function(x, ...) {
  return("classification")
}

# Setup lime::predict_model() function for keras
predict_model.lgb.Booster <- function(x, newdata, type, ...) {
  pred <- predict(x, data.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}

# Test our predict_model() function
predict_model(x = lgb, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime(
  x              = x_train_tbl, 
  model          = lgb, 
  bin_continuous = T,
  n_bins = 10
  )

# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[1,], 
  explainer    = explainer, 
  n_labels     = 1,
  n_features   = 10,
  kernel_width = 0.5)


plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

# Feature correlations to Churn
corrr_analysis <- x_train_tbl %>%
  mutate(Churn = y_train_vec) %>%
  correlate() %>%
  focus(Churn) %>%
  rename(feature = rowname) %>%
  arrange(abs(Churn)) %>%
  mutate(feature = as_factor(feature)) 
corrr_analysis

# Correlation visualization
corrr_analysis %>%
  ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
  geom_point() +
  # Positive Correlations - Contribute to churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[2]], 
               data = corrr_analysis %>% filter(Churn > 0)) +
  geom_point(color = palette_light()[[2]], 
             data = corrr_analysis %>% filter(Churn > 0)) +
  # Negative Correlations - Prevent churn
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[1]], 
               data = corrr_analysis %>% filter(Churn < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = corrr_analysis %>% filter(Churn < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  theme_tq() +
  labs(title = "Churn Correlation Analysis",
       subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
       y = "Feature Importance")
