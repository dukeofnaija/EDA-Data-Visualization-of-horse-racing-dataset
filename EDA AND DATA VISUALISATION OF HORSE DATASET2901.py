#!/usr/bin/env python
# coding: utf-8

# # Loading Dataset into Jupyter notebook

# In[1]:


#importing libraries necessary for the exercise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df_horse = pd.read_csv('Horses.csv') #loading dataset into a dataframe


# # EXPLORATORY DATA ANALYSIS

# In[3]:


df_horse.head() #revealing the first then records


# In[5]:


#some columns are truncated and will require full view to see full data

pd.set_option('display.max_columns', None)
df_horse.head() #first 10 rows of the untruncated data


# # Quick observations
# * Startingprice, ForecastPrice contain date inputs
# * ScheduleTime contains invalid inputs ####

# # Initial explorations

# In[6]:


df_horse.info()   # This gives us understanding of the dataset to see number of records, datatypes, non-null values count


# In[7]:


df_horse.isnull().sum() #This displays the dataset and the sum total of the null values by column.


# # Secondary Observations
# 
# From the dataset above, we can observe a few problems already <br>
# * The columns Hood, Eyeshield,Eyecover,TongueStrap and CheekPiece are completely null with no values and may need to be dropped.
# * OverweightValue, Visor have very significant null values
# * Meeting date has got the wrong data type(object instead of datetime), StartingPrice and ForecastValue(object instead of float) <br>
# 
# # Questions
# 1. Why are there so many null values and empty columns?
# 2. Could it be that the data has not been properly warehoused or enteries not done by a Data analyst who understands the implications of the errors?

# In[8]:


df_horse.describe()  #statistics for numerical columns


# # Statistical Analyses <br>
# ### <font color=black> The statistical analyses shows some of the horse specific attributes in the data </font>
# * The interquartile range or middle 50% of the horses are aged between 3 and 5 years old with the oldest 10 years old
# * The horses have weight value between 101 and 148 with the average weightValue 126.6
# 

# In[9]:


df_horse.nunique() #helps us to identify categorical columns and how many categories.


# The Won column (target variable) is binary and seems appropriate for analysis.
# This could also be important for predictive analyses/classification modelling if there is a need to.

# ### Correlation test
#     * It will be interesting to see how the individual features correlate and especially against the Won column

# In[12]:


#A minimal cleaning of the dataset would be necessary before visualization.
columns_to_drop = ['Hood','Eyecover','EyeShield','CheekPieces','TongueStrap','ScheduledTime']# list of colums with entirely null values
df_horse=df_horse.drop(columns=columns_to_drop, errors='ignore') # drop columns with null values


# In[13]:


#heatmap for numeric features to understand correlation of the variables with the win
plt.figure(figsize=(10, 6))
correlation = df_horse.corr()
sns.heatmap(correlation,annot=False, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap', fontsize=14)
plt.show()


# ### Observation from correlation
# * A few correlations exist between some of the numerical features
# * No feature has a strong correlation with the target variable Won
# 

# # DATA VISUALIZATION

# #### Pie Chart to determine the percentage mix of winnings and losses

# In[15]:


won_pct = df_horse['Won'].value_counts() #returns a series with index of 0s and 1s along with their count

# Create pie chart
plt.pie(won_pct, labels=won_pct.index, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart of Categories for win')
plt.show()


# only 8% of the data contains winnings.
# Note: The dataset is imbalanced, with more horses not winning (Won = 0) compared to those that did (Won = 1).
# Insight: in the event of predictive modelling or supervised learning classification, we might encounter challenges like Overfitting and low sensititivity to wins.

# #### Box and whisker Plot to determine how the weight values 

# In[16]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_horse, x='Won', y='WeightValue', palette='pastel')

# Add title and labels
plt.title('Distribution of Weight Values by Winning Outcome', fontsize=14)
plt.xlabel('Winning Outcome (0 = loss, 1 = Win)', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# ### Observations
# * From the boxplot, there is no significant difference between the median of the weight value for Win and loss. <br>
# * A narrower box for Won = 1 suggests more consistent weight values for winning horses as against the wider box for won=0 indicating variability in the weights of losing horses.
# * significant overlap between the weightValue distribution for the win and loss might suggest that weight alone is not a strong predictor of winning
# * More outliers exist for the wins than losses. Does this imply that horses with significantly light weight could still win?

# ### Time series to observe the trend of races against the number of races

# In[20]:


#plotting a time series to understand the race trends with time

df_horse['MeetingDate'] = pd.to_datetime(df_horse['MeetingDate']) # changing the MeetingDate columnt to datetime datatype

# Group by date and count the number of races
races_over_time = df_horse.groupby('MeetingDate').size()

# Plot races over time
plt.figure(figsize=(12, 6))
races_over_time.plot(kind='line', color='black')
plt.title('Number of Races Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Races', fontsize=12)
plt.grid(alpha=0.5)
plt.show()


# ### Observations
# * In the first few months of the year(2017,2018), the data suggests that there is a spike in the number or races.
# * In 2018 the data suggests the trend was relatively stable until the year ended
# * After the major spike In February 2018, The number of races trended downwards with a big deep(less than 50) in July of 2018
# * What could have happened in 2018 that led to a reduced number of races without usual relative consistency? This is worth investigating as it may have serious commercial implications.

# ### Observing features and how they could affect winning.

# In[21]:


df_horse_win = df_horse[df_horse['Won']==1] #filtering the dataframe for wins
df_horse_win.head()


# ### How does WeightValue affect Winnings

# In[22]:


df_horse_win['WeightValue'].hist(bins=20,edgecolor='Black')

#plot histogram of weightvalue against number of wins
plt.title('Winner by weight value')
plt.xlabel('weight value')
plt.ylabel('Number of wins')


# ### observations <br>
# * The data is left skewed or negatively skewed with the horses with higher weights contributing to the most wins.
# * Horses with weight value of 132 had the most wins.

# ### How many races each horse has won.

# In[23]:


# Group by 'HorseID' and count the number of wins
horse_wins = df_horse_win.groupby('HorseID').size().reset_index(name='WinCount') #reset_index() converts the series to Dataframe

# Plot a histogram of win counts
plt.figure(figsize=(12, 6))
plt.hist(horse_wins['WinCount'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Horse Wins', fontsize=14)
plt.xlabel('Number of Wins', fontsize=12)
plt.ylabel('Number of Horses', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ### Observations
# * A significant number of horses(over 1400) won only once and only very few had consistently high wins.<br>
# can we infer from the data that the races are very competitive or the horses are having inconsistent performance?

# ### Investigating to what extent external factors affected or contributed to the horse winning.
# 

# In[26]:


# Count occurrences for Weather and Track Type
grouped_external_factors = df_horse_win.groupby(['Weather', 'TrackType']).size().unstack(fill_value=0)

# Plotting a bar chart
grouped_external_factors.plot(kind='bar', stacked=True, figsize=(8, 5), color=['skyblue', 'gold', 'green', 'orange'],edgecolor='black')

# Adding labels and title
plt.title('Number of Wins by Weather Condition (Categorized by TrackType)')
plt.xlabel('Weather')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.legend(title='TrackType')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # SUMMARY AND QUESTIONS
# 1. Dataset consists of several missing values which brings the integrity of the data to question.
# 2. Wrong or unexpected values in ForecastPrice,StartingPrice ans ScheduledTime columns.
# 3. How would the above observation affect the overall analyses? This could be very important features that can aid our analysis.
# 3. Imbalance in Target column Won.There seems to be more losses than wins making the data not exactly suitable for predictive modelling.
# 4. From the dataset, success of a horse could be affected by external conditions like weather and Tracktype.
# 5. Why is it that only 2 years of this data was provided? would this be sufficient to make conclusive analyses on trends?
# 6. Horses seemed to have several sexes g,f,m,c,h. Could this be actually gender's for horses or error in input?
# 
# # CHALLENGES
# 1. Time constraints- The time allotted for this exercise was not sufficient for a more detailed analysis.
# 2. No domain knowledge on Horse racing. No data dictionary was provided to enable understanding of most of the features.

# In[ ]:




