#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Student Name : Sheethal Melnarse
# Cohort       : 3 (Castro)

################################################################################
# Import Packages
################################################################################

import numpy as np
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
import sklearn.model_selection 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
import sklearn.model_selection
import random as rand # random number generation

################################################################################
# Load Data
################################################################################

file = 'C:/Users/SHEETHAL/Downloads/Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)

# Renaming the file name

my_chef = original_df


################################################################################
# Feature Engineering 
################################################################################

# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in my_chef.iterrows():
    
    # splitting email domain at '@'
    split_email = my_chef.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame

# safety measure in case of multiple concatenations
original_df = pd.read_excel(file)

my_chef = original_df

# renaming column to concatenate
email_df.columns = ['NAME' , 'EMAIL_DOMAIN']


# concatenating personal_email_domain with friends DataFrame
my_chef = pd.concat([my_chef, email_df['EMAIL_DOMAIN']],
                   axis = 1)

# printing value counts of personal_email_domain
my_chef.loc[: ,'EMAIL_DOMAIN'].value_counts()

# email domain types
personal_domains = ['@gmail.com', '@protonmail.com', '@yahoo.com']
professional_domains  = ['@mmm.com','@amex.com','@apple.com','@boeing.com','@caterpillar.com','@chevron.com','@cisco.com','@cocacola.com','@disney.com','@dupont.com','@exxon.com','@ge.org','@goldmansacs.com','@homedepot.com','@ibm.com','@intel.com','@jnj.com','@jpmorgan.com','@mcdonalds.com','@merck.com','@microsoft.com','@nike.com','@pfizer.com','@pg.com','@travelers.com','@unitedtech.com','@unitedhealth.com','@verizon.com','@visa.com','@walmart.com']
junk_domains = ['@me.com','@aol.com','@hotmail.com','@live.com','@msn.com','@passport.com']

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in my_chef['EMAIL_DOMAIN']:
    if '@' + domain in personal_domains:
        placeholder_lst.append('personal')
        
    elif '@' + domain in professional_domains:
        placeholder_lst.append('professional')
    
    elif '@' + domain in junk_domains:
        placeholder_lst.append('junk')
        
    else:
        print('Unknown')


# concatenating with original DataFrame
my_chef['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

# Imputing the NULL values with 'Unknown'
my_chef['FAMILY_NAME'].fillna(value='Unknown')
my_chef = my_chef.drop('NAME', axis = 1)
my_chef = my_chef.drop('EMAIL', axis = 1)
my_chef = my_chef.drop('FIRST_NAME', axis = 1)

# Outlier threshold

TOTAL_MEALS_ORDERED_HI = 120
UNIQUE_MEALS_PURCH_HI = 7
CONTACTS_W_CUSTOMER_SERVICE_HI = 9
PRODUCT_CATEGORIES_VIEWED_HI = 6
AVG_TIME_PER_SITE_VISIT_HI = 160
CANCELLATIONS_BEFORE_NOON_HI = 4
CANCELLATIONS_BEFORE_NOON_LO = 0
CANCELLATIONS_AFTER_NOON_HI = 2
AVG_PREP_VID_TIME_HI = 230
FOLLOWED_RECOMMENDATIONS_PCT_HI = 40
AVG_CLICKS_PER_VISIT_HI = 16
TOTAL_PHOTOS_VIEWED_LO = 60
LARGEST_ORDER_SIZE_HI = 6
WEEKLY_PLAN_HI = 19
WEEKLY_PLAN_LO = 0
EARLY_DELIVERIES_HI = 5
EARLY_DELIVERIES_LO = 1
LATE_DELIVERIES_HI = 7
MASTER_CLASSES_ATTENDED_HI = 1.5

## REVENUE
REVENUE_HI = 2100

# developing features (columns) for outliers

# TOTAL_MEALS_ORDERED
my_chef['OUT_TOTAL_MEALS_ORDERED'] = 0
condition_hi = my_chef.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][my_chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

my_chef['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# UNIQUE_MEALS_PURCH
my_chef['OUT_UNIQUE_MEALS_PURCH'] = 0
condition_hi = my_chef.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][my_chef['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HI]

my_chef['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
my_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = my_chef.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][my_chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]

my_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# PRODUCT_CATEGORIES_VIEWED
my_chef['OUT_PRODUCT_CATEGORIES_VIEWED'] = 0
condition_hi = my_chef.loc[0:,'OUT_PRODUCT_CATEGORIES_VIEWED'][my_chef['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_HI]

my_chef['OUT_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_TIME_PER_SITE_VISIT
my_chef['OUT_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][my_chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HI]

my_chef['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_BEFORE_NOON
my_chef['OUT_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = my_chef.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][my_chef['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_HI]
condition_lo = my_chef.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][my_chef['CANCELLATIONS_BEFORE_NOON'] < CANCELLATIONS_BEFORE_NOON_LO]

my_chef['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_AFTER_NOON
my_chef['OUT_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = my_chef.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][my_chef['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

my_chef['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_PREP_VID_TIME
my_chef['OUT_AVG_PREP_VID_TIME'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_PREP_VID_TIME'][my_chef['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]

my_chef['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# FOLLOWED_RECOMMENDATIONS_PCT
my_chef['OUT_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition_hi = my_chef.loc[0:,'OUT_FOLLOWED_RECOMMENDATIONS_PCT'][my_chef['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_HI]

my_chef['OUT_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# EARLY_DELIVERIES
my_chef['OUT_EARLY_DELIVERIES'] = 0
condition_hi = my_chef.loc[0:,'OUT_EARLY_DELIVERIES'][my_chef['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]
condition_lo = my_chef.loc[0:,'OUT_EARLY_DELIVERIES'][my_chef['EARLY_DELIVERIES'] < EARLY_DELIVERIES_LO]

my_chef['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# AVG_CLICKS_PER_VISIT
my_chef['OUT_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][my_chef['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]

my_chef['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# TOTAL_PHOTOS_VIEWED
my_chef['OUT_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = my_chef.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][my_chef['TOTAL_PHOTOS_VIEWED'] < TOTAL_PHOTOS_VIEWED_LO]

my_chef['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# LARGEST_ORDER_SIZE
my_chef['OUT_LARGEST_ORDER_SIZE'] = 0
condition_hi = my_chef.loc[0:,'OUT_LARGEST_ORDER_SIZE'][my_chef['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]

my_chef['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# WEEKLY_PLAN
my_chef['OUT_WEEKLY_PLAN'] = 0
condition_hi = my_chef.loc[0:,'OUT_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]
condition_lo = my_chef.loc[0:,'OUT_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] < WEEKLY_PLAN_LO]

my_chef['OUT_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_WEEKLY_PLAN'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# LATE_DELIVERIES
my_chef['OUT_LATE_DELIVERIES'] = 0
condition_hi = my_chef.loc[0:,'OUT_LATE_DELIVERIES'][my_chef['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

my_chef['OUT_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# MASTER_CLASSES_ATTENDED
my_chef['OUT_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = my_chef.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][my_chef['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

my_chef['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# REVENUE
my_chef['OUT_REVENUE'] = 0
condition_hi = my_chef.loc[0:,'OUT_REVENUE'][my_chef['REVENUE'] > REVENUE_HI]

my_chef['OUT_REVENUE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# setting trend-based thresholds
# HIGH
TOTAL_MEALS_ORDERED_CHANGE_HI = 140 # data scatters above this point
CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI = 10.5 # data scatters above this point
AVG_TIME_PER_SITE_VISIT_CHANGE_HI = 200 # data scatters above this point
CANCELLATIONS_BEFORE_NOON_CHANGE_HI = 5 # data scatters above this point
AVG_PREP_VID_TIME_CHANGE_HI = 250 # data scatters above this point
TOTAL_PHOTOS_VIEWED_CHANGE_HI = 400 # data scatters above this point
LARGEST_ORDER_SIZE_CHANGE_HI = 8 # data scatters above this point
WEEKLY_PLAN_CHANGE_HI = 14 # data scatters above this point
LATE_DELIVERIES_CHANGE_HI = 7.5 # data scatters above this point


# LOW
AVG_CLICKS_PER_VISIT_CHANGE_LO = 11 # data scatters below this point
LARGEST_ORDER_SIZE_CHANGE_LO = 2 # data scatters below this point

# AT SPECIFIC POINT
UNIQUE_MEALS_PURCH_CHANGE_AT = 1 # only changes at 1
CANCELLATIONS_AFTER_NOON_CHANGE_AT = 0 # zero inflated
TOTAL_PHOTOS_VIEWED_CHANGE_AT = 0 # zero inflated
WEEKLY_PLAN_CHANGE_AT = 0 # zero inflated
MEDIAN_MEAL_RATING_CHANGE_AT = 3

##############################################################################
## Feature Engineering (trend changes)                                      ##
##############################################################################

########################################
## change above threshold                ##
########################################

# greater than sign

# TOTAL_MEALS_ORDERED
my_chef['CHANGE_TOTAL_MEALS_ORDERED'] = 0
condition = my_chef.loc[0:,'CHANGE_TOTAL_MEALS_ORDERED'][my_chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_CHANGE_HI]

my_chef['CHANGE_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
my_chef['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition = my_chef.loc[0:,'CHANGE_CONTACTS_W_CUSTOMER_SERVICE'][my_chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_CHANGE_HI]

my_chef['CHANGE_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# AVG_TIME_PER_SITE_VISIT
my_chef['CHANGE_AVG_TIME_PER_SITE_VISIT'] = 0
condition = my_chef.loc[0:,'CHANGE_AVG_TIME_PER_SITE_VISIT'][my_chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_CHANGE_HI]

my_chef['CHANGE_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# CANCELLATIONS_BEFORE_NOON
my_chef['CHANGE_CANCELLATIONS_BEFORE_NOON'] = 0
condition = my_chef.loc[0:,'CHANGE_CANCELLATIONS_BEFORE_NOON'][my_chef['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_CHANGE_HI]

my_chef['CHANGE_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# AVG_PREP_VID_TIME
my_chef['CHANGE_AVG_PREP_VID_TIME'] = 0
condition = my_chef.loc[0:,'CHANGE_AVG_PREP_VID_TIME'][my_chef['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_CHANGE_HI]

my_chef['CHANGE_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# TOTAL_PHOTOS_VIEWED
my_chef['CHANGE_TOTAL_PHOTOS_VIEWED'] = 0
condition = my_chef.loc[0:,'CHANGE_TOTAL_PHOTOS_VIEWED'][my_chef['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_CHANGE_HI]
condition_a = my_chef.loc[0:,'CHANGE_TOTAL_PHOTOS_VIEWED'][my_chef['TOTAL_PHOTOS_VIEWED'] == TOTAL_PHOTOS_VIEWED_CHANGE_AT]

my_chef['CHANGE_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)
my_chef['CHANGE_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_a,
                                   value      = 1,
                                   inplace    = True)


# LARGEST_ORDER_SIZE
my_chef['CHANGE_LARGEST_ORDER_SIZE'] = 0
condition = my_chef.loc[0:,'CHANGE_LARGEST_ORDER_SIZE'][my_chef['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_CHANGE_HI]
condition_l = my_chef.loc[0:,'CHANGE_LARGEST_ORDER_SIZE'][my_chef['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_CHANGE_LO]

my_chef['CHANGE_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)
my_chef['CHANGE_LARGEST_ORDER_SIZE'].replace(to_replace = condition_l,
                                   value      = 1,
                                   inplace    = True)


# LATE_DELIVERIES
my_chef['CHANGE_LATE_DELIVERIES'] = 0
condition = my_chef.loc[0:,'CHANGE_LATE_DELIVERIES'][my_chef['LATE_DELIVERIES'] > LATE_DELIVERIES_CHANGE_HI]

my_chef['CHANGE_LATE_DELIVERIES'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

########################################
## change below threshold                ##
########################################

# lower than sign

# AVG_CLICKS_PER_VISIT
my_chef['CHANGE_AVG_CLICKS_PER_VISIT'] = 0
condition = my_chef.loc[0:,'CHANGE_AVG_CLICKS_PER_VISIT'][my_chef['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_CHANGE_LO]

my_chef['CHANGE_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)


########################################
## change at threshold                ##
########################################

# double-equals sign

# UNIQUE_MEALS_PURCH
my_chef['CHANGE_UNIQUE_MEALS_PURCH'] = 0
condition = my_chef.loc[0:,'CHANGE_UNIQUE_MEALS_PURCH'][my_chef['UNIQUE_MEALS_PURCH'] == UNIQUE_MEALS_PURCH_CHANGE_AT]

my_chef['CHANGE_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# CANCELLATIONS_AFTER_NOON
my_chef['CHANGE_CANCELLATIONS_AFTER_NOON'] = 0
condition = my_chef.loc[0:,'CHANGE_CANCELLATIONS_AFTER_NOON'][my_chef['CANCELLATIONS_AFTER_NOON'] == CANCELLATIONS_AFTER_NOON_CHANGE_AT]

my_chef['CHANGE_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# WEEKLY_PLAN
my_chef['CHANGE_WEEKLY_PLAN'] = 0
condition = my_chef.loc[0:,'CHANGE_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] == WEEKLY_PLAN_CHANGE_AT]
condition_h = my_chef.loc[0:,'CHANGE_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] > WEEKLY_PLAN_CHANGE_HI]

my_chef['CHANGE_WEEKLY_PLAN'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)
my_chef['CHANGE_WEEKLY_PLAN'].replace(to_replace = condition_h,
                                   value      = 1,
                                   inplace    = True)

# MEDIAN_MEAL_RATING
my_chef['CHANGE_MEDIAN_MEAL_RATING'] = 0
condition = my_chef.loc[0:,'CHANGE_MEDIAN_MEAL_RATING'][my_chef['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_CHANGE_AT]

my_chef['CHANGE_MEDIAN_MEAL_RATING'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)

# one hot encoding categorical variables
ONE_HOT_MEDIAN_MEAL_RATING = pd.get_dummies(my_chef['MEDIAN_MEAL_RATING'])

# dropping categorical variables after they've been encoded
my_chef = my_chef.drop('MEDIAN_MEAL_RATING', axis = 1)

# joining codings together
my_chef = my_chef.join([ONE_HOT_MEDIAN_MEAL_RATING])

# saving feature-rich dataset in Excel
my_chef.to_excel('my_chef_feature_rich.xlsx',
                 index = False)


################################################################################
# Train/Test Split
################################################################################


# applying model in scikit-learn

# declaring set of x-variables
x_variables = ['CROSS_SELL_SUCCESS','TOTAL_MEALS_ORDERED','CONTACTS_W_CUSTOMER_SERVICE','AVG_TIME_PER_SITE_VISIT',
              'CANCELLATIONS_BEFORE_NOON','MOBILE_LOGINS','LATE_DELIVERIES','FOLLOWED_RECOMMENDATIONS_PCT',
              'AVG_PREP_VID_TIME','LARGEST_ORDER_SIZE','MASTER_CLASSES_ATTENDED','AVG_CLICKS_PER_VISIT','TOTAL_PHOTOS_VIEWED',
              'OUT_TOTAL_MEALS_ORDERED','OUT_UNIQUE_MEALS_PURCH','OUT_CONTACTS_W_CUSTOMER_SERVICE',
              'OUT_AVG_PREP_VID_TIME','OUT_FOLLOWED_RECOMMENDATIONS_PCT','OUT_AVG_CLICKS_PER_VISIT',
              'OUT_LARGEST_ORDER_SIZE','OUT_WEEKLY_PLAN','OUT_LATE_DELIVERIES','CHANGE_CONTACTS_W_CUSTOMER_SERVICE','CHANGE_LATE_DELIVERIES',
              'CHANGE_UNIQUE_MEALS_PURCH','CHANGE_WEEKLY_PLAN','CHANGE_CANCELLATIONS_AFTER_NOON','CHANGE_AVG_CLICKS_PER_VISIT',
              'OUT_AVG_TIME_PER_SITE_VISIT','EARLY_DELIVERIES',1,2,4,5]

# Preparing a DataFrame based the the analysis above
my_chef_data   = my_chef.loc[ : , x_variables]

# preparing response variable
my_chef_target = my_chef.loc[:,'REVENUE']


# running train/test split again
X_train, X_test, y_train, y_test = train_test_split(
                                    my_chef_data,
                                    my_chef_target,
                                    test_size = 0.25,
                                    random_state = 222)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a model object
ridge_model = sklearn.linear_model.Ridge()

# FITTING the training data
ridge_fit  = ridge_model.fit(X_train, y_train)


# PREDICTING on new data
ridge_pred = ridge_fit.predict(X_test)


################################################################################
# Final Model Score (score)
################################################################################

print('test_score:',  ridge_model.score(X_test, y_test).round(3))

