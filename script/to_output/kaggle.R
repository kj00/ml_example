##
source("script/input_to_intermidiate/prepare.R")

# Load libraries
Sys.setenv(PATH = "C:/Users/Koji/Anaconda3")

library(keras)
library(corrr)

##

# Building our Artificial Neural Network

create_model <- 
  function(drop_p, input_shape) {
    model_keras <- keras_model_sequential()
    
    model_keras %>% 
      # First hidden layer
      layer_batch_normalization(
        input_shape = input_shape) %>% 
      layer_dense(
        units              = 64, 
        kernel_initializer = "uniform", 
        activation         = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = drop_p) %>%
      layer_dense(
        units              = 64, 
        kernel_initializer = "uniform", 
        activation         = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = drop_p) %>%
      layer_dense(
        units              = 64, 
        kernel_initializer = "uniform", 
        activation         = "relu") %>% 
      layer_batch_normalization() %>% 
      layer_dropout(rate = drop_p) %>%
      # Second hidden layer
      layer_dense(
        units              = 64, 
        kernel_initializer = "uniform", 
        activation         = "relu") %>% 
      # Dropout to prevent overfitting
      layer_batch_normalization() %>% 
      layer_dropout(rate = drop_p) %>%
      # Output layer
      layer_dense(
        units              = 1, 
        kernel_initializer = "uniform", 
        activation         = "sigmoid")
    # Compile ANN
    return(model_keras)}

model_keras <- create_model(0.5, ncol(x_train_tbl))

# Fit the keras model to the training data
compile(model_keras,
        optimizer = optimizer_adam(lr = 0.000001),
        loss      = 'binary_crossentropy',
        metrics   = c('accuracy')
)

fit_keras <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 6000, 
  epochs           = 10,
  validation_data = list(as.matrix(x_test_tbl), y_test_vec)
)

#Psuedo labeling
yhat_keras_class_vec <- predict_classes(object = model_keras,
                                        x = as.matrix(x_test_tbl)) %>%
  as.vector()

fit_keras <- fit(
  object           = model_keras, 
  x                = rbind(as.matrix(x_train_tbl),
                           as.matrix(x_test_tbl)), 
  y                = c(y_train_vec, yhat_keras_class_vec),
  validation_data =list(as.matrix(x_test_tbl), y_test_vec),
  batch_size       = 4048*2, 
  epochs           = 30)

# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras,
                                        x = as.matrix(x_test_tbl)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras,
                                      x = as.matrix(x_test_tbl)) %>%
  as.vector()
hist(yhat_keras_prob_vec)

# Format test data and predictions for yardstick metrics
library(forcats)
estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
) %>% 
  arrange(-class_prob) %>% 
  mutate(test = 
           (purrr::accumulate(estimate == "yes" & truth == "yes", `+`) +
              purrr::accumulate(estimate == "no" & truth == "yes", `+`))/ sum(truth == "yes"))


estimates_keras_tbl

# Confusion Table
estimates_keras_tbl %>% conf_mat(truth, estimate)

# Accuracy
estimates_keras_tbl %>% metrics(truth, estimate)
# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)
#loglos
estimates_l2_tbl %>%
  mutate(class_prob_no = 1-class_prob) %>% 
  mnLogLoss(truth, class_prob, class_prob_no)/nrow(test_tbl)

# Precision
tibble(
  precision = estimates_keras_tbl %>% precision(truth, estimate),
  recall    = estimates_keras_tbl %>% recall(truth, estimate)
)

# Setup lime::model_type() function for keras
model_type.keras.models.Sequential <- function(x, ...) {
  return("classification")
}


# Setup lime::predict_model() function for keras
predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  return(data.frame(Yes = pred, No = 1 - pred))
}

# Test our predict_model() function
predict_model(x = model_keras, newdata = x_test_tbl, type = 'raw') %>%
  tibble::as_tibble()

# Run lime() on training set
explainer <- lime::lime(
  x              = x_train_tbl, 
  model          = model_keras, 
  bin_continuous = FALSE)


# Run explain() on explainer
explanation <- lime::explain(
  x_test_tbl[1:10,], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 4,
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
