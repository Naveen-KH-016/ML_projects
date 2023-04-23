# Jesus is my Saviour!
import os
os.chdir('C:\\Users\\Admin\\Desktop\\backup_files\\LMS_ML_PROJCTS')
import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder 

df = pd.read_csv('h1n1_vaccine_prediction.csv') #26707; 34 , 1st is unique id 
df.info()
# lets drop 1st unique_id 
df33 = df.drop(['unique_id'], axis = 1)
df33.info() # now 33 columns
# droping all will give 11794 rows only 50% values 
dfnomissing = df33.dropna()
dfnomissing.info() # # 11,794 rows; 33 columns
# its not a good idea to carry with 11,794 rows
## note that index identification remains same as in 
# the original file 
'''
so, we will remove the column- "has_health_insur" (14433 rows only)
and create a new file name 'df32'. This is shown below. 
'''
df32 = df33.drop(['has_health_insur'], axis = 1)
df32.info() # observe many missing values in many columns, now 32 columns

# lets remove all missing values from df32
df_vac = df32.dropna() # 19642 , 32 
df_vac.info()
## now , 19642 is a good no to go with! 

###__________________1 ALWAYS START WITH TARGET VARIABLE
# 1 h1n1_vaccine - Target Variable
df_vac.h1n1_vaccine.isnull().sum() #No Missing values
df_vac.h1n1_vaccine.value_counts() 
'''
0    15128
1     4514
Name: h1n1_vaccine, dtype: int64
'''
# Bar Plot
sns.countplot(x = 'h1n1_vaccine', data = df_vac , palette = 'hot')
plt.title('Barplot of h1n1_vaccine')
#__________________ 2 h1n1_worry [0,1,2,3] ordered
#______histogram
#_run in block
plt.hist(df_vac.h1n1_worry, bins = 'auto', facecolor = 'red')
plt.xlabel('h1n1_worry')
plt.ylabel('counts')
plt.title('Histogram of h1n1_worry') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['h1n1_worry'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.h1n1_worry.isnull().sum() #0 Missing values
df_vac.h1n1_worry.value_counts() 
'''
: 
2.0    7989
1.0    6229
3.0    3175
0.0    2249
Name: h1n1_worry, dtype: int64'''

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_worry ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
# 1.564e-79 ie p_value is <0.05; Ho Reject; Good Predictor

# let's judge from chisquare way!
from scipy.stats import chi2_contingency
ct_worry = pd.crosstab(df_vac.h1n1_vaccine, df_vac.h1n1_worry)
chi2_contingency(ct_worry, correction = False)
# p_val = 4.9e-78, Ho reject, hence association exists, good predictor 

#_____________________________# 3 h1n1_awareness [0,1,2] ordered
#______histogram
#_run in block
plt.hist(df_vac.h1n1_awareness, bins = 'auto', facecolor = 'red')
plt.xlabel('h1n1_awareness')
plt.ylabel('counts')
plt.title('Histogram of h1n1_awareness') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['h1n1_awareness'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.h1n1_awareness.isnull().sum() #0 Missing values
df_vac.h1n1_awareness.value_counts() 
'''
1.0    10861
2.0     7362
0.0     1419'''

# Bar Plot
sns.countplot(x = 'h1n1_awareness', data = df_vac , palette = 'spring')
plt.title('Countplot of h1n1_awareness')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('h1n1_awareness ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#3.442e-70 ie  p_value is <0.05; Ho Reject; Good Predictor

#_______________ 4 antiviral_medication [0 and 1]

df_vac.antiviral_medication.isnull().sum() #0 Missing values
df_vac.antiviral_medication.value_counts() 
'''
0.0    18671
1.0      971'''

# Bar Plot
sns.countplot(x = 'antiviral_medication', data = df_vac , palette = 'cool')
plt.title('Countplot of antiviral_medication')

#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_antiviral = pd.crosstab(df_vac.h1n1_vaccine, df_vac.antiviral_medication)
chi2_contingency(ct_antiviral, correction = False)
# p_val = 3.9e-7, Ho reject, hence association exists, good predictor 

#________________ 5 contact_avoidance [0 and 1]

df_vac.contact_avoidance.isnull().sum() #0 Missing values
df_vac.contact_avoidance.value_counts() 
'''
1.0    14544
0.0     5098'''
# Bar Plot
sns.countplot(x = 'contact_avoidance', data = df_vac , palette = 'turbo')
plt.title('Countplot of contact_avoidance')

#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_avoid = pd.crosstab(df_vac.h1n1_vaccine, df_vac.contact_avoidance)
chi2_contingency(ct_avoid, correction = False)
# p_val = 6.6 e-10, Ho reject, hence association exists, good predictor 

#_________________ 6 bought_face_mask [0 and 1]

df_vac.bought_face_mask.isnull().sum() #0 Missing values
df_vac.bought_face_mask.value_counts() 
'''
0.0    18312
1.0     1330'''
# Bar Plot
sns.countplot(x = 'bought_face_mask', data = df_vac , palette = 'afmhot')
plt.title('Countplot of bought_face_mask')
#Hypothesis Testing
ct_mask = pd.crosstab(df_vac.h1n1_vaccine, df_vac.bought_face_mask)
chi2_contingency(ct_mask, correction = False)
# p_val = 4.9 e-26, Ho reject, hence association exists, good predictor 

#________________ 7 wash_hands_frequently [0 and 1] 

df_vac.wash_hands_frequently.isnull().sum() #0 Missing values
df_vac.wash_hands_frequently.value_counts() 
'''
1.0    16399
0.0     3243'''
# Bar Plot
sns.countplot(x = 'wash_hands_frequently', data = df_vac , palette = 'rainbow')
plt.title('Countplot of wash_hands_frequently')

#Hypothesis Testing
ct_wash = pd.crosstab(df_vac.h1n1_vaccine, df_vac.wash_hands_frequently)
chi2_contingency(ct_wash, correction = False)
# p_val = 4.3 e-26, Ho reject, hence association exists, good predictor 

#__________________ 8 avoid_large_gatherings [0 and 1]

df_vac.avoid_large_gatherings.isnull().sum() #0 Missing values
df_vac.avoid_large_gatherings.value_counts() 
'''
0.0    12703
1.0     6939'''
# Bar Plot
sns.countplot(x = 'avoid_large_gatherings', data = df_vac , palette = 'icefire')
plt.title('Countplot of avoid_large_gatherings')
#Hypothesis Testing
ct_gath = pd.crosstab(df_vac.h1n1_vaccine, df_vac.avoid_large_gatherings)
chi2_contingency(ct_gath, correction = False)
# p_val = 0.004, Ho reject, hence association exists, good predictor 

#______________ 9 reduced_outside_home_cont [0 and 1] 
df_vac.reduced_outside_home_cont.isnull().sum() #0 Missing values
df_vac.reduced_outside_home_cont.value_counts() 
'''
0.0    13159
1.0     6483'''
# Bar Plot
sns.countplot(x = 'reduced_outside_home_cont', data = df_vac , palette = 'gist_heat')
plt.title('Countplot of reduced_outside_home_cont')
#Hypothesis Testing
ct_outside = pd.crosstab(df_vac.h1n1_vaccine, df_vac.reduced_outside_home_cont)
chi2_contingency(ct_outside, correction = False)
# p_val = 0.015, Ho reject, hence association exists, good predictor 

#__________________10 avoid_touch_face [0 and 1]
df_vac.avoid_touch_face.isnull().sum() #0 Missing values
df_vac.avoid_touch_face.value_counts() 
'''
1.0    13455
0.0     6187'''
# Bar Plot
sns.countplot(x = 'avoid_touch_face', data = df_vac , palette = 'GnBu')
plt.title('Countplot of avoid_touch_face')
#Hypothesis Testing
ct_face = pd.crosstab(df_vac.h1n1_vaccine, df_vac.avoid_touch_face)
chi2_contingency(ct_face, correction = False)
# p_val = 1.5e-23, Ho reject, hence association exists, good predictor 

#_____________________ 11 dr_recc_h1n1_vacc [0 and 1]
df_vac.dr_recc_h1n1_vacc.isnull().sum() #0 Missing values
df_vac.dr_recc_h1n1_vacc.value_counts() 
'''
0.0    15203
1.0     4439'''
# Bar Plot
sns.countplot(x = 'dr_recc_h1n1_vacc', data = df_vac , palette = 'Reds')
plt.title('Histogram of dr_recc_h1n1_vacc')
#Hypothesis Testing
ct_drrec = pd.crosstab(df_vac.h1n1_vaccine, df_vac.dr_recc_h1n1_vacc)
chi2_contingency(ct_drrec, correction = False)
# p_val = 0, Ho reject, hence association exists, good predictor 


#_______________ 12 dr_recc_seasonal_vacc [0 and 1]
df_vac.dr_recc_seasonal_vacc.isnull().sum() #0 Missing values
df_vac.dr_recc_seasonal_vacc.value_counts()
'''
0.0    13091
1.0     6551'''
# Bar Plot
sns.countplot(x = 'dr_recc_seasonal_vacc', data = df_vac , palette = 'mako')
plt.title('Barplot of dr_recc_seasonal_vacc')
#Hypothesis Testing
ct_drseason = pd.crosstab(df_vac.h1n1_vaccine, df_vac.dr_recc_seasonal_vacc)
chi2_contingency(ct_drseason, correction = False)
# p_val = 2.2e-192, Ho reject, hence association exists, good predictor 

#________________________ 13 chronic_medic_condition [0 and 1]
df_vac.chronic_medic_condition.isnull().sum() #0 Missing values
df_vac.chronic_medic_condition.value_counts() 
'''
0.0    14066
1.0     5576'''
# Bar Plot
sns.countplot(x = 'chronic_medic_condition', data = df_vac , palette = 'Dark2')
plt.title('Barplot of chronic_medic_condition')
#Hypothesis Testing
ct_chronic = pd.crosstab(df_vac.h1n1_vaccine, df_vac.chronic_medic_condition)
chi2_contingency(ct_chronic, correction = False)
# p_val = 1.39e-49, Ho reject, hence association exists, good predictor 

#_________________ 14 cont_child_undr_6_mnths [0 and 1]
df_vac.cont_child_undr_6_mnths.isnull().sum() #0 Missing values
df_vac.cont_child_undr_6_mnths.value_counts() 
'''
0.0    17995
1.0     1647'''
# Bar Plot
sns.countplot(x = 'cont_child_undr_6_mnths', data = df_vac , palette = 'Accent')
plt.title('Barplot of cont_child_undr_6_mnths')
#Hypothesis Testing
ct_child = pd.crosstab(df_vac.h1n1_vaccine, df_vac.cont_child_undr_6_mnths)
chi2_contingency(ct_child, correction = False)
# p_val = 9.2e-26, Ho reject, hence association exists, good predictor 

#____________________ 15 is_health_worker [0 and 1]
df_vac.is_health_worker.isnull().sum() #0 Missing values
df_vac.is_health_worker.value_counts() 
'''
0.0    17310
1.0     2332'''
# Bar Plot
sns.countplot(x = 'is_health_worker', data = df_vac , palette = 'Set2')
plt.title('Countplot of is_health_worker')
#Hypothesis Testing
ct_hw = pd.crosstab(df_vac.h1n1_vaccine, df_vac.is_health_worker)
chi2_contingency(ct_hw, correction = False)
# p_val = 4e-152, Ho reject, hence association exists, good predictor 

#____________________ 16 is_h1n1_vacc_effective [1,2,3,4,5] ordered
#______histogram
#_run in block
plt.hist(df_vac.is_h1n1_vacc_effective, bins = 'auto', facecolor = 'red')
plt.xlabel('is_h1n1_vacc_effective')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_vacc_effective') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_h1n1_vacc_effective'].plot.box(color=props2, patch_artist = True, vert = False)
# few outliers on lower side; IGNORE! 
df_vac.is_h1n1_vacc_effective.isnull().sum() #0 Missing values
df_vac.is_h1n1_vacc_effective.value_counts() 
'''
4.0    9172
5.0    5715
3.0    2838
2.0    1347
1.0     570'''

# Bar Plot
sns.countplot(x = 'is_h1n1_vacc_effective', data = df_vac , palette = 'Dark2')
plt.title('Countplot of is_h1n1_vacc_effective')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('is_h1n1_vacc_effective~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.0  ie p_value which is <0.05; Ho Reject; Good Predictor

#_______________________ 17 is_h1n1_risky [1,2,3,4,5] ordered
#______histogram
#_run in block
plt.hist(df_vac.is_h1n1_risky, bins = 'auto', facecolor = 'red')
plt.xlabel('is_h1n1_risky')
plt.ylabel('counts')
plt.title('Histogram of is_h1n1_risky') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_h1n1_risky'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.is_h1n1_risky.isnull().sum() #0 Missing values
df_vac.is_h1n1_risky.value_counts() 
'''
2.0    7691
1.0    5881
4.0    4184
5.0    1348
3.0     538'''

# Bar Plot
sns.countplot(x = 'is_h1n1_risky', data = df_vac , palette = 'Set1')
plt.title('Barplot of is_h1n1_risky')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('is_h1n1_risky ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.0  ie p_value is <0.05; Ho Reject; Good Predictor

#____________________ 18 sick_from_h1n1_vacc [1,2,3,4,5] ordered 
#______histogram
#_run in block
plt.hist(df_vac.sick_from_h1n1_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('sick_from_h1n1_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_h1n1_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['sick_from_h1n1_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.sick_from_h1n1_vacc.isnull().sum() #0 Missing values
df_vac.sick_from_h1n1_vacc.value_counts() 
'''
2.0    6956
1.0    6684
4.0    4390
5.0    1560
3.0      52'''

# Bar Plot
sns.countplot(x = 'sick_from_h1n1_vacc', data = df_vac , palette = 'turbo')
plt.title('Countplot of sick_from_h1n1_vacc')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('sick_from_h1n1_vacc ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#3.62e-31  ie p_value is <0.05; Ho Reject; Good Predictor

#_______________ 19 is_seas_vacc_effective [1,2,3,4,5] ordered
#______histogram
#_run in block
plt.hist(df_vac.is_seas_vacc_effective, bins = 'auto', facecolor = 'red')
plt.xlabel('is_seas_vacc_effective')
plt.ylabel('counts')
plt.title('Countplot of is_seas_vacc_effective') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_seas_vacc_effective'].plot.box(color=props2, patch_artist = True, vert = False) 
# few are on lower side; Ignore outliers

df_vac.is_seas_vacc_effective.isnull().sum() #0 Missing values
df_vac.is_seas_vacc_effective.value_counts() 
'''
4.0    8906
5.0    7603
2.0    1638
1.0     822
3.0     673'''
# Bar Plot
sns.countplot(x = 'is_seas_vacc_effective', data = df_vac , palette = 'turbo')
plt.title('Barplot of is_seas_vacc_effective')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('is_seas_vacc_effective ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#9.2e-152 ie p_value is <0.05; Ho Reject; Good Predictor

df_vac.info()
#_____________________ 20 is_seas_risky [1,2,3,4,5] ordered 
#______histogram
#_run in block
plt.hist(df_vac.is_seas_risky, bins = 'auto', facecolor = 'red')
plt.xlabel('is_seas_risky')
plt.ylabel('counts')
plt.title('Histogram of is_seas_risky') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['is_seas_risky'].plot.box(color=props2, patch_artist = True, vert = False) 
# no outliers 
df_vac.is_seas_risky.isnull().sum() #0 Missing values
df_vac.is_seas_risky.value_counts() 
'''
2.0    6811
4.0    5984
1.0    4258
5.0    2286
3.0     303'''
# Bar Plot
sns.countplot(x = 'is_seas_risky', data = df_vac , palette = 'turbo')
plt.title('Barplot of is_seas_risky')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('is_seas_risky ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.0 ie p_value which is <0.05; Ho Reject; Good Predictor

#___________________ 21 sick_from_seas_vacc [1,2,3,4,5] ordered
#______histogram
#_run in block
plt.hist(df_vac.sick_from_seas_vacc, bins = 'auto', facecolor = 'red')
plt.xlabel('sick_from_seas_vacc')
plt.ylabel('counts')
plt.title('Histogram of sick_from_seas_vacc') 
#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['sick_from_seas_vacc'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

df_vac.sick_from_seas_vacc.isnull().sum() #0 Missing values
df_vac.sick_from_seas_vacc.value_counts() 
'''
1.0    8996
2.0    5713
4.0    3683
5.0    1221
3.0      29'''
# Bar Plot
sns.countplot(x = 'sick_from_seas_vacc', data = df_vac , palette = 'turbo')
plt.title('Barplot of sick_from_seas_vacc')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('sick_from_seas_vacc ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.36 ie p_value is >0.05; Ho accepted; Bad Predictor

#_____________ 22 age_bracket [actually ordered]
#______histogram
#_run in block
plt.hist(df_vac.age_bracket, bins = 'auto', facecolor = 'red')
plt.xlabel('age_bracket')
plt.ylabel('counts')
plt.title('Histogram of age_bracket') 

#____boxplot
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['age_bracket'].plot.box(color=props2, patch_artist = True, vert = False) 
## oops, its object, lets change to 1,2,3 like; first see value counts
df_vac.age_bracket.isnull().sum() #0 Missing values
df_vac.age_bracket.value_counts() 
'''
65+ Years        4491
55 - 64 Years    4234
45 - 54 Years    4038
18 - 34 Years    3925
35 - 44 Years    2954
'''
# let categories be in order

df_vac['age_bracket'] =df_vac.get('age_bracket').replace('65+ Years', 5)
df_vac['age_bracket'] =df_vac.get('age_bracket').replace('55 - 64 Years', 4)
df_vac['age_bracket'] =df_vac.get('age_bracket').replace('45 - 54 Years', 3)
df_vac['age_bracket'] =df_vac.get('age_bracket').replace('18 - 34 Years', 1)
df_vac['age_bracket'] =df_vac.get('age_bracket').replace('35 - 44 Years', 2)
# ignore warnings ! 

df_vac.age_bracket.isnull().sum() #0 Missing values
df_vac.age_bracket.value_counts()
'''
5    4491
4    4234
3    4038
1    3925
2    2954
Name: age_bracket, dtype: int64
'''
# now boxplot will come
props2 = dict(boxes = 'red', whiskers = 'green', medians = 'black', caps = 'red')
df_vac['age_bracket'].plot.box(color=props2, patch_artist = True, vert = False) #No outliers

#__________________________we could have followed a more easier way! 
# label encoding the data ; its good for nominal data , not good for ordered data
# like in our present case!
# DO NOT TRY AS IT HAS ALREAY BEING DONE !!!!
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
#df_vac['age_bracket']= le.fit_transform(df_vac['age_bracket']) 
df_vac.age_bracket.value_counts()
'''
5    4491
4    4234
3    4038
1    3925
2    2954'''
#____________________________________________________________________

# Bar Plot
sns.countplot(x = 'age_bracket', data = df_vac , palette = 'Dark2')
plt.title('Barplot of age_bracket')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('age_bracket ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#1.5e-10 ie p_value is <0.05; Ho rejected; Good Predictor

#__________________ 23 qualification - object, Actually ordered! 3 levels
df_vac.qualification.isnull().sum() 
df_vac.qualification.value_counts()
'''
College Graduate    8165
Some College        5570
12 Years            4287
< 12 Years          1620'''

# let's put them in order
df_vac['qualification'] =df_vac.get('qualification').replace('College Graduate', 4)
df_vac['qualification'] =df_vac.get('qualification').replace('Some College', 3)
df_vac['qualification'] =df_vac.get('qualification').replace('12 Years', 2)
df_vac['qualification'] =df_vac.get('qualification').replace('< 12 Years', 1)

df_vac.qualification.value_counts()
'''
4    8165
3    5570
2    4287
1    1620
'''
# Bar Plot
sns.countplot(x = 'qualification', data = df_vac , palette = 'Set2')
plt.title('Barplot of qualification')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('qualification ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#1.57e-22 ie p_value is <0.05; Ho rejected; Good Predictor

#_________________ 24 race - object [NO ORDER, NOMINAL]; 4 levels
df_vac.race.isnull().sum() #No Missing values
df_vac.race.value_counts() 
'''
White                15745
Black                 1474
Hispanic              1295
Other or Multiple     1128'''

# label encoding 'race' ; does alphabetically! 
# HERE WE CAN USE LabelEncoder!
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['race']= le.fit_transform(df_vac['race']) 
df_vac.race.value_counts()
'''
3    15745
0     1474
1     1295
2     1128'''

# Bar Plot
sns.countplot(x = 'race', data = df_vac , palette = 'Dark2')
plt.title('Countplot of race')

#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_race = pd.crosstab(df_vac.h1n1_vaccine, df_vac.race)
chi2_contingency(ct_race, correction = False)
# p_val = 2.4e-10, Ho reject, hence association exists, good predictor 

#___________________ 25 sex - object [female, male]
df_vac.sex.isnull().sum() #No Missing values
df_vac.sex.value_counts() 
'''
Female    11638
Male       8004'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['sex']= le.fit_transform(df_vac['sex']) 
df_vac.sex.value_counts()
'''
0    11638
1     8004'''
# Bar Plot
sns.countplot(x = 'sex', data = df_vac , palette = 'Set1')
plt.title('Countplot of sex')

#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_sex = pd.crosstab(df_vac.h1n1_vaccine, df_vac.sex)
chi2_contingency(ct_sex, correction = False)
# p_val = 00, Ho reject, hence association exists, good predictor 

#______________ 26 income_level - object, its ordered
df_vac.income_level.isnull().sum() # no missing values 
df_vac.income_level.value_counts()
'''
<= $75,000, Above Poverty    11185
> $75,000                     6159
Below Poverty                 2298'''

#Converting to numeric/ integer
df_vac['income_level']=df_vac.get('income_level').replace('Below Poverty', 1)
df_vac['income_level']=df_vac.get('income_level').replace('<= $75,000, Above Poverty', 2)
df_vac['income_level']=df_vac.get('income_level').replace('> $75,000', 3)

df_vac.income_level.value_counts()
'''
2    11185
3     6159
1     2298
'''
df_vac.info()

# Bar Plot
sns.countplot(x = 'income_level', data = df_vac , palette = 'Dark2')
plt.title('Countplot of income_level')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('income_level ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
# 2.62e-15 ie p_value is <0.05; Ho rejected; Good Predictor

#___________________27 marital_status - object [0,1]
df_vac.marital_status.isnull().sum() #471 Missing values
df_vac.marital_status.value_counts() 
'''
Married        10768
Not Married     8874'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['marital_status']= le.fit_transform(df_vac['marital_status']) 
df_vac.marital_status.value_counts()
'''
0    10768
1     8874'''
df_vac.info()

# Bar Plot
sns.countplot(x = 'marital_status', data = df_vac , palette = 'winter')
plt.title('Countplot of marital_status')

#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_mari = pd.crosstab(df_vac.h1n1_vaccine, df_vac.marital_status)
chi2_contingency(ct_mari, correction = False)
# p_val = 2.14e-13, Ho reject, hence association exists, good predictor 

#__________28 housing_status - object [own, rent]
df_vac.housing_status.isnull().sum()
df_vac.housing_status.value_counts() 
'''
Own     14980
Rent     4662'''
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()   
df_vac['housing_status']= le.fit_transform(df_vac['housing_status']) 
df_vac.housing_status.value_counts()
'''
0    14980
1     4662'''
# Bar Plot
sns.countplot(x = 'housing_status', data = df_vac , palette = 'Set2')
plt.title('Barplot of housing_status')
#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_house = pd.crosstab(df_vac.h1n1_vaccine, df_vac.housing_status)
chi2_contingency(ct_house, correction = False)
# p_val = 7.1e-07, Ho reject, hence association exists, good predictor 

# _______________ 29 employment - object [3 levels] actually ordered
df_vac.employment.isnull().sum() 
df_vac.employment.value_counts() 
'''
Employed              11093
Not in Labor Force     7417
Unemployed             1132'''

#Converting to numeric/ integer
df_vac['employment']=df_vac.get('employment').replace('Employed', 3)
df_vac['employment']=df_vac.get('employment').replace('Not in Labor Force', 2)
df_vac['employment']=df_vac.get('employment').replace('Unemployed', 1)

df_vac.employment.value_counts() 
'''
3    11093
2     7417
1     1132
'''
#____________________LabelEncoder giving opposite notation!
# label encoding the data ; DO NOT TRY THIS
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['employment']= le.fit_transform(df_vac['employment']) 
df_vac.employment.value_counts()
'''
0    11093
1     7417
2     1132'''
#________________________________________________
# Bar Plot
sns.countplot(x = 'employment', data = df_vac , palette = 'Reds')
plt.title('Barplot of employment')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('employment ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.015 ie p_value is <0.05; Ho rejected; Good Predictor

#____________30 census_msa - object, 3 levels, NOMINAL
df_vac.census_msa.isnull().sum() #No Missing values
df_vac.census_msa.value_counts() 
'''
MSA, Not Principle  City    8571
MSA, Principle City         5717
Non-MSA                     5354'''

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
  
df_vac['census_msa']= le.fit_transform(df_vac['census_msa']) 
df_vac.census_msa.value_counts()
'''
0    8571
1    5717
2    5354'''

# Bar Plot
sns.countplot(x = 'census_msa', data = df_vac , palette = 'Dark2')
plt.title('Countplot of census_msa')
#Hypothesis Testing
from scipy.stats import chi2_contingency
ct_msa = pd.crosstab(df_vac.h1n1_vaccine, df_vac.census_msa)
ct_msa
chi2_contingency(ct_msa, correction = False)
# p_val = 0.76, > 0.05 Ho accept, hence association does not exists, bad predictor 

#______________31 no_of_adults, ordered  
df_vac.no_of_adults.isnull().sum() #No Missing values
df_vac.no_of_adults.value_counts() 
'''
1.0    11006
0.0     5683
2.0     2124
3.0      829'''

# Bar Plot
sns.countplot(x = 'no_of_adults', data = df_vac , palette = 'Set2')
plt.title('Countplot of no_of_adults')

#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('no_of_adults ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.55 ie p_value is >0.05; Ho accepted; Bad Predictor

#_______________ 32 no_of_children , ordered 
df_vac.no_of_children.isnull().sum() #No Missing values
df_vac.no_of_children.value_counts() 
'''
0.0    13697
1.0     2402
2.0     2207
3.0     1336'''
# Bar Plot
sns.countplot(x = 'no_of_children', data = df_vac , palette = 'Dark2')
plt.title('Countplot of no_of_children')
#Hypothesis Testing
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('no_of_children ~ h1n1_vaccine', data = df_vac).fit()
aov_table = sm.stats.anova_lm(mod)
print(aov_table)
#0.63 ie p_value is >0.05; Ho accepted; Bad Predictor

#++++++++++++++++++++++++
df_vac.info()
'''
lets delete
index 19, sick_from_seas_vacc
index 28, census_msa
index 29, no_of_adults
index 30, no_of_children

AND SAVE NEW DATA AS hn and export to wd and 
THEN START A NEW SCRIPT
'''
hn = df_vac.drop(['sick_from_seas_vacc','census_msa','no_of_adults','no_of_children'], axis = 1)
hn.info() # 19642, 28 columns 

hn.to_csv('hn.csv')

#________lets create dummy variables for 'race'

df2 = pd.get_dummies(hn.race, drop_first = True, prefix = 'race')

hnd = pd.concat([hn, df2], axis = 1)

# we must remove the original col 'race'

hnc = hnd.drop(['race'], axis = 1)
hnc.info() #19642, 30 columns 

hnc.to_csv('hnc.csv')
