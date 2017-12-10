library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(tidyverse)
library(data.table)
library(dtplyr)

##
churn_data_id <- fread("data/input/train.csv")
churn_data_tr <- fread("data/input/transactions.csv")

churn_data_kg_sample <- churn_data_id %>%
  sample_frac(size = 0.1) %>%
  merge(., churn_data_tr, by = "msno")


rm(churn_data_id, churn_data_tr)
glimpse(churn_data_kg_sample)

## agg
churn_data_kg_sample <- unique(churn_data_kg_sample)

churn_data_kg_sample_p1 <- churn_data_kg_sample[, .(msno,
                                                    payment_method_id)] %>%
  unique %>%
  
  dcast(msno ~ payment_method_id, fill = 0) %>%
  map_if(is.integer, ~ (.x !=0) %>%
           as.integer()) %>%
  as.data.table()

churn_data_kg_sample_p2 <- 
churn_data_kg_sample[, -"payment_method_id"] %>% 
  .[,
                     .(Churn = if_else(sum(is_churn)==1, "Yes", "No"),
                       payment_plan_days_mean = mean(payment_plan_days),
                       actual_amount_paid_sum = sum(actual_amount_paid, na.rm = T),
                       is_auto_renew_mean = mean(is_auto_renew),
                       transaction_date_max = max(transaction_date),
                       transaction_count = length(transaction_date),
                       membership_expire_date = max(membership_expire_date),
                       is_cancel_mean = mean(is_cancel)
                     ),
                     by = msno]

churn_data_kg <- merge(churn_data_kg_sample_p1,
                       churn_data_kg_sample_p2,
                       by = "msno")
churn_data_kg$msno %>% duplicated() %>% sum

##
# Remove unnecessary data
churn_data_tbl <- churn_data_kg %>%
  select(-msno) %>%
  drop_na() %>%
  select(Churn, everything())

glimpse(churn_data_tbl)

##
# Split test/training sets
#set.seed(100)
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_test_split

##
# Retrieve train and test sets
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)


##
rec_obj <- recipe(Churn ~ ., data = train_tbl) %>%
  prep(data = train_tbl)

##
# Predictors
x_train_tbl <- bake(rec_obj, newdata = train_tbl) %>% select(-Churn)
x_test_tbl  <- bake(rec_obj, newdata = test_tbl) %>% select(-Churn)

# Response variables for training and testing sets
y_train_vec <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test_vec  <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)
