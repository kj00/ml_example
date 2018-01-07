#

### return table of result in tidy way

ml_result <- function(y_test_vec,
                      pred,
                      class_levels = 1:0,
                      class_names = c("yes", "no")) {
  stopifnot(length(y_test_vec) != length(pred))
  stopifnot(length(class_levels) == length(class_names) &
              length(class_levels)== 2)
  
  ## format factor levels
  truth <-
    y_test_vec %>%
    factor(levels = class_levels)
  levels(truth) <- class_names
  
  guess <-
    as.integer(pred >= 0.5) %>%
    factor(levels = class_levels)
  levels(guess) <- class_names
  
  
  ## summarize result
  guess_tbl <- tibble::tibble(truth      =  truth,
                              guess = guess,
                              class_prob = pred) %>%
    dplyr::arrange(-class_prob) %>%
    dplyr::mutate(
      acc_tp = purrr::accumulate(guess == class_names[1] &
                                   truth == class_names[1],
                                 `+`),
      acc_fn = purrr::accumulate(guess == class_names[2] &
                                   truth == class_names[1], `+`),
      gain = (acc_tp + acc_fn) / sum(truth == class_names[1])
    )
  
  
  return(guess_tbl)
}


# return evaluation metrics from ml result table
ml_evaluate <- function(guess_tbl,
                        metric = c("conf_mat",
                                   "accuracy",
                                   "auc",
                                   "logloss",
                                   "precision_recall")) {
  
  
  stopifnot(is_tibble(guess_tbl))
  stopifnot(!all(c("truth", "guess", "class_prob") %in%
                   colnames(guess_tbl)))
  
  
  cal_metric <-
    list(
      "conf_mat" = function(x)
        yardstick::conf_mat(x, truth, guess),
      "accuracy" = function(x)
        yardstick::metrics(x, truth, guess),
      "auc" = function(x)
        yardstick::roc_auc(x, truth, class_prob),
      "logloss" = function(x)
        dplyr::mutate(x, class_prob_no = 1 - class_prob) %>%
        yardstick::mnLogLoss(truth, class_prob, class_prob_no),
      "precision_recall" =  function(x)
        tibble::tibble(
          precision = x %>%  yardstick::precision(truth, guess),
          recall    = x %>% yardstick::recall(truth, guess)
        )
    )
  
  cal_metric <- cal_metric[metric]
  
  out <- invoke_map(cal_metric, x = guess_tbl)
  return(out)
}


##

plot(
  y = guess_tbl$test,
  x = seq_len(length(guess_tbl$test)) / length(guess_tbl$test),
  type = "l"
)
lines(x = seq(0, 1, 0.1),
      y = seq(0, 1, 0.1))
plot(y = estimates_lg_tbl$test,
     x = seq_len(length(estimates_lg_tbl$test)) / length(estimates_lg_tbl$test))