
# coding: utf-8

# # HMDA analysis
# 
# ## Getting data

# In[1]:

# imports
import pandas as pd
import numpy as np
import urllib,json


# The dataset have been browsed in order to understand its contents. The fields are defined at http://cfpb.github.io/api/hmda/fields.html and there is a numerical index for every property. The fields with numerical indices have been selected to be saved. For the race, only the first reported race for applicant and co-applicant has been saved.

# In[2]:

collumns=['action_taken',
 'agency_code',
 'applicant_ethnicity',
 'applicant_income_000s',
 'applicant_race_1',
 'applicant_sex',
 'application_date_indicator',
 'co_applicant_ethnicity',
 'co_applicant_race_1',
 'co_applicant_sex',
 'county_code',
 'edit_status',
 'hoepa_status',
 'hud_median_family_income',
 'lien_status',
 'loan_amount_000s',
 'loan_purpose',
 'minority_population',
 'msamd',
 'number_of_1_to_4_family_units',
 'number_of_owner_occupied_units',
 'population',
 'preapproval',
 'property_type',
 'purchaser_type',
 'respondent_id',
 'sequence_number',
 'tract_to_msamd_income']


# Using the GUI to download csv file using filters defined in https://www.consumerfinance.gov/data-research/hmda/explore#!/as_of_year=2015&state_code-1=6&property_type=1,2&owner_occupancy=1&section=filters. The selection is
#    * state California
#    * year 2015
#    * property type: 1-4 family dwelling and manufactured houses
#    * owner-occupied principal residence

# In[3]:

df=pd.read_csv('/home/jpavel/Downloads/hmda_lar (1).csv')


# Select only numerical columns

# In[4]:

df=df[collumns]


# The goal is to find which applications were approved, which means the actions 1 ("loan originated") and 2 ("approved but not accepted"). Therefore a new field giving a binary value approved/not approved was created

# In[5]:

df['approved']=(df['action_taken']<3)


# Replace NaN by zero, as it is actually an information (no quality problem)

# In[6]:

df['edit_status']=df['edit_status'].fillna(0)


# Now split the dataset for those with defined income and those without

# In[7]:

df['hasDefinedIncome']=df['applicant_income_000s'].notnull()


# The strategy for replacing the missing values will be:
#     1) for categorical ones - the missing value will be a category
#     2) for continuous ones - create another variable describing if variable is defined or not, and withing the variable replace by mean to do not change the mean of the variable

# In[8]:

def fillNAwithMean(colname):
    df[str('hasDefined'+colname)]=df[colname].notnull()
    mean=df[colname].mean()
    df[colname]=df[colname].fillna(mean)


# In[9]:

fillNAwithMean('applicant_income_000s')


# In[10]:

df['county_code']=df['county_code'].fillna(0)


# In[11]:

df['hasDefinedHUDMedianIncome']=df['hud_median_family_income'].notnull()


# In[12]:

fillNAwithMean('hud_median_family_income')


# In[13]:

df['hasDefined_minority_population']=df['minority_population'].notnull()


# In[14]:

fillNAwithMean('minority_population')


# In[15]:

df['msamd']=df['msamd'].fillna(0)


# In[16]:

fillNAwithMean('number_of_1_to_4_family_units')


# In[17]:

fillNAwithMean('number_of_owner_occupied_units')


# In[18]:

fillNAwithMean('population')


# In[19]:

fillNAwithMean('tract_to_msamd_income')


# Now there are no missing values in the data frame. Let's transform categories into dummy variables:

# In[24]:

# dropped variables with >10 categories
vars_cat=['agency_code','application_date_indicator','applicant_ethnicity','applicant_race_1','applicant_sex','co_applicant_ethnicity','co_applicant_race_1',
          'co_applicant_sex','edit_status','hoepa_status','lien_status','loan_purpose','preapproval',
          'property_type','purchaser_type'
         ]


# Truncating unnecessary columns - action_taken (transferred to approved), owner_occupance (used for data slice selection), sequence_number (random application number)

# In[25]:

cols_remove=['action_taken','sequence_number']
def deleteCols(inDF,cols):
    outDF=inDF
    for col in cols:
        del outDF[col]
    
    return outDF


# In[26]:

df_pruned=df


# In[27]:

df_pruned=deleteCols(df_pruned,cols_remove)


# In[28]:

df_cat=pd.get_dummies(df_pruned,columns=vars_cat)


# In[31]:

deleteCols(df_cat,['msamd','county_code','respondent_id'])


# Before embarking on model search, let's have a look on variable distribution:

# ## Model building
# 
# Now start the classfication. First create feature matrix and targed vector.

# In[33]:

X=df_cat.drop('approved', axis=1)


# In[37]:

y=df_cat['approved']


# ### Random Forest Classifier
# Train random Forest classifier

# In[38]:

from sklearn.cross_validation import train_test_split


# In[39]:

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=1)


# In[40]:

from sklearn.metrics import accuracy_score


# In[43]:

from sklearn.ensemble import RandomForestClassifier


# In[44]:

clf=RandomForestClassifier(n_estimators=200, max_features=None, n_jobs=2,random_state=1,verbose=1)


# In[ ]:

clf.fit(Xtrain,ytrain)


# test it
y_model = model.predict(Xtest)
accuracy_score(ytest,y_model)




