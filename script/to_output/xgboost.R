##
source("script/input_to_intermidiate/prepare.R")

##
library(xgboost)

##
bst <- xgb.train(
  data = xgb.DMatrix(data.matrix(x_train_tbl),
                     label = y_train_vec),
  watchlist = list(validation =
                     xgb.DMatrix(data.matrix(x_test_tbl),
                                 label = y_test_vec)),
  nrounds = 30000,
  params = list(
    max_depth = 1,
    eta = 0.005,
    lambda = 1,
    nthread = 2,
    subsample = 0.8,
    colsample_bytree = 0.5,
    tree_method = "approx",
    grow_policy = "lossguide",
    objective = "binary:logistic",
    eval_metric = "error"
  ),
  print_every_n = 10,
  early_stopping_rounds = 1000
)


#
bst <- xgb.load("xgboost.model")
pred <- predict(bst, data.matrix(x_test_tbl),
                ntreelimit = bst$best_ntreelimit)


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
plot(y = estimates_lg_tbl$test,
     x = seq_len(length(estimates_lg_tbl$test)) / length(estimates_lg_tbl$test))



##
# Run lime() on training set
explainer <- lime::lime(x              = x_train_tbl,
                        model          = bst,
                        bin_continuous = FALSE)

xgboost::xgb.ggplot.deepness(bst)
xgboost::(bst)

##
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)

for (i in seq_along(importance_matrix$Feature)) {
  importance_matrix$Feature[i] <- colnames(x_train_tbl)[i]
}

xgb.plot.importance(importance_matrix = importance_matrix)

