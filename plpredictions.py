#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from scipy.stats import poisson,skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[22]:


df = pd.read_csv('pl_final.csv')
df = df[['HomeTeam','AwayTeam','FTHG','FTAG']]
df = df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
df


# In[23]:


# why we use poisson distibution ?
# Its a discrete probability function that descirbes the probability within a specific amount of time (here it is 90 mins)


# In[24]:


goals_data = pd.concat([df[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           df[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goals_data, 
                        family=sm.families.Poisson()).fit()
#poisson_model.summary()


# In[25]:


#to calculate the winning percentage of home and away teams



def matrix(mat,max_goals):
    upper_sum = 0
    
    
    for i in range((max_goals)):
        for j in range((max_goals)):
            if (i<=j):
                upper_sum += mat[i][j]
                
    return upper_sum
                
def matrix2(mat,max_goals):
    lower_sum  = 0
    
    for i in range((max_goals)):
        for j in range((max_goals)):
            if (i>=j):
                lower_sum += mat[i][j]
                
    return lower_sum
    
    

    
            
    
    


# In[26]:


def answer(homeTeam , awayTeam  , max_goals):
    
    home_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = poisson_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    
    a = [[poisson.pmf(i,team_avg)for i in range(0,int(max_goals)+1)]for team_avg in [home_goals_avg,away_goals_avg]]
    
    scores = np.outer(np.array(a[0]),np.array(a[1]))
    
    home_team_win = matrix2(scores,max_goals) - np.sum(np.diag(scores))
    away_team_win = matrix(scores,max_goals) - np.sum(np.diag(scores))
    
    
    print("The probability of a draw is {:.2f}".format(np.sum(np.diag(scores))))
    print("The probability of {} winning is {:.2f}".format(homeTeam,home_team_win))
    print("The probability of {} winning is {:.2f}".format(awayTeam,away_team_win))
    
    
    accuracy = np.sum(np.diag(scores)) + home_team_win + away_team_win
    print("The accuracy of result is: {:.2f}".format(accuracy))
   
    
    return  




# In[27]:


homeTeam = input("Enter home team: ")
awayTeam = input("Enter away team: ")
max_goals = int(input("Enter the maximum number of goals 1 team can score: "))

answer(homeTeam , awayTeam , max_goals)


# In[ ]:





# In[ ]:





# In[ ]:




