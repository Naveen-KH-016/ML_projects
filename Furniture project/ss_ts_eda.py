# Jesus is my saviour!! 

import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.utils import resample
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
#%matplotlib inline
from math import sqrt
#from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_column',None)

df = pd.read_csv('Super_Store_ts.csv', encoding= 'latin1') #2121; 21 , 1st is RowID 
df.info()

#Changing the datatype being its a date
df['Order_Date'] = pd.to_datetime(df['OrderDate'])
df['Order_year'] = pd.DatetimeIndex(df['OrderDate']).year
df['Order_month'] = pd.DatetimeIndex(df['OrderDate']).month
df.info()
df.shape #2121, 24
df.Order_Date.head()
df.Order_Date.describe()


# sales vs Order_Date [after index, not work! ]
plt.plot(df.Order_Date, df.Sales) # difficult to interpret!
plt.xticks(rotation=90)

# order date vs ship date
df.info()
df.Ship Date = pd.to_datetime(df.Ship Date) #not worked!
# rename Ship Date to ShipDate
df = df.rename(columns = {'Ship Date': 'ShipDate'})
df.ShipDate = pd.to_datetime(df.ShipDate) # worked!haha
df.info()

df.ShipDate.describe()
'''
count                    2121
unique                    960
top       2015-12-16 00:00:00
freq                       10
first     2014-01-10 00:00:00
last      2018-01-05 00:00:00
Name: ShipDate, dtype: object
'''

# nos of days in delivery?
df['DaysInDelivery'] = df['ShipDate'].sub(df['Order_Date'], axis=0)
df['DaysInDelivery'].head() 

df.DaysInDelivery.describe()
'''
count                         2121
mean     3 days 22:00:30.551626591
std      1 days 18:07:32.231661933
min                0 days 00:00:00
25%                3 days 00:00:00
50%                4 days 00:00:00
75%                5 days 00:00:00
max                7 days 00:00:00
Name: DaysInDelivery, dtype: object '''


#Indexing data with Order_Date
df = df.set_index('Order_Date')
df.head()


# year wise
plt.plot(df.groupby('Order_year')['Sales'].count())
plt.plot(df.groupby('Order_year')['Sales'].sum())
plt.plot(df.groupby('Order_year')['Sales'].mean())
plt.plot(df.groupby('Order_year')['Sales'].median())

#___________let's see monthwise sales in different way!
#Resample helps to filter data sec, min, hour, day, week, month, year wise 
#Resample works only when index is in date format..
plt.plot(df.Sales.resample('M').sum(), 'r')
plt.xticks(rotation=45)
plt.title('Sales monthwise')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.show()


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#1 Sales
df.Sales.describe()
'''
count    2121.000000
mean      349.834887
std       503.179145
min         1.892000
25%        47.040000
50%       182.220000
75%       435.168000
max      4416.174000
Name: Sales, dtype: float64
'''
df.Sales.value_counts() #less useful here
sns.distplot(df.Sales) #good plot
sns.boxplot(df.Sales) # good insight 

#2 ShipMode

df.ShipMode.describe() #useful
df.ShipMode.value_counts()
'''
Standard Class    1248
Second Class       427
First Class        327
Same Day           119
Name: ShipMode, dtype: int64 '''

sns.countplot(df.ShipMode, palette = 'bright')

# Sum Sales by ShipMode 
plt.plot(df.groupby('ShipMode')['Sales'].sum())
plt.plot(df.groupby('ShipMode')['Sales'].count())

#3 Customer Name
df.info()
df.Customer Name.unique().shape # not worked
df = df.rename(columns = {'Customer Name': 'CustomerName'})
df.CustomerName.unique().shape
df.CustomerName.unique() #too big list, haha!

# RowId, OrderID, Customer ID, Postal Codes,
# ProductID, Quantity, Discount, Profit; seem to be less
# important for time series analysis

#4 segment 
df.info()
df.Segment.describe() # useful
df.Segment.value_counts()
'''
Consumer       1113
Corporate       646
Home Office     362
Name: Segment, dtype: int64
'''

sns.countplot(df.Segment, palette = 'Dark2')

# Sum Sales by Segment
plt.plot(df.groupby('Segment')['Sales'].sum())
plt.plot(df.groupby('Segment')['Sales'].count())

#5 Country
df.info()
df.Country.describe() #not so useful, only 1 country
df.Country.value_counts()
'''
count              2121
unique                1
top       United States
freq               2121
Name: Country, dtype: object
'''

sns.countplot(df.Country, palette = 'Dark2')

#6 city
df.info()
df.City.describe() #useful
'''
count              2121
unique              371
top       New York City
freq                192
Name: City, dtype: object
'''
df.City.value_counts()
'''
New York City    192
Los Angeles      154
Philadelphia     111
San Francisco    102
Seattle           97

Bowling Green      1
Bryan              1
Nashua             1
Mission Viejo      1
Sioux Falls        1
Name: City, Length: 371, dtype: int64

'''
sns.countplot(x = df.City, palette = 'turbo')

# sum Sales by City
plt.plot(df.groupby('City')['Sales'].sum())
plt.plot(df.groupby('City')['Sales'].count())

citysales = df.groupby('City')['Sales'].sum()
citysales.head()
#take this [citysales] to desktop 
# from variable explorer
# or, export through code

cs = pd.DataFrame(citysales)
cs.to_csv('cs.csv')

#..........or
css = cs.sort_values(by="Sales", ascending=False) # highest, 1st
# now take this to desktop for further analysis

#7 State
df.info()
df.State.describe() #useful
df.State.value_counts()
'''
California              444
New York                236
Texas                   202

Wyoming                   1
Montana                   1
Maine                     1
Name: State, dtype: int64
'''

sns.countplot(x = df.State, palette = 'Set2')

# sum Sales by state
plt.plot(df.groupby('State')['Sales'].sum())
plt.plot(df.groupby('State')['Sales'].count())

statesales = df.groupby('State')['Sales'].sum()
statesales.head()
#take this [statesales] to desktop 
# from variable explorer
# or, export through code

ss = pd.DataFrame(statesales)
ss.to_csv('ss.csv')

#..........or
ss = ss.sort_values(by="Sales", ascending=False) # highest, 1st
# now take this to desktop for further analysis

#8 Region
df.info()
df.Region.describe() #useful
df.Region.value_counts()
'''
West       707
East       601
Central    481
South      332
Name: Region, dtype: int64
'''

sns.countplot(x = df.Region, palette = 'Set2')

# sum Sales by Region
plt.plot(df.groupby('Region')['Sales'].sum())
plt.plot(df.groupby('Region')['Sales'].count())

regionsales = df.groupby('Region')['Sales'].sum()
regionsales.head()
#take this [regionsales] to desktop 
# from variable explorer
# or, export through code

rs = pd.DataFrame(regionsales)
rs.to_csv('rs.csv')

#..........or
rs = rs.sort_values(by="Sales", ascending=False) # highest, 1st
# now take this to desktop for further analysis

#9 Category
df.info()
df.Category.describe() #NOT useful
df.Category.value_counts()
'''
count          2121
unique            1
top       Furniture
freq           2121
Name: Category, dtype: object
'''
#10 SubCategory
df.info()
df.SubCategory.describe() #useful
df.SubCategory.value_counts()
'''
df.SubCategory.describe() #useful
Out[248]: 
count            2121
unique              4
top       Furnishings
freq              957
Name: SubCategory, dtype: object

df.SubCategory.value_counts()
Out[249]: 
Furnishings    957
Chairs         617
Tables         319
Bookcases      228
Name: SubCategory, dtype: int64
'''

sns.countplot(x = df.SubCategory, palette = 'Set2')

# sum Sales by SubCategory
plt.plot(df.groupby('SubCategory')['Sales'].sum())
plt.plot(df.groupby('SubCategory')['Sales'].count())

#11 ProductName
df.info()
df.ProductName.describe() #useful
df.ProductName.value_counts() #big list
'''
count                           2121
unique                           380
top       KI Adjustable-Height Table
freq                              18
Name: ProductName, dtype: object
'''

sns.countplot(x = df.ProductName, palette = 'rainbow') #lessuseful

# sum Sales by ProductName
plt.plot(df.groupby('ProductName')['Sales'].sum())
plt.plot(df.groupby('ProductName')['Sales'].count())

psales = df.groupby('ProductName')['Sales'].sum()
psales.head()
#take this [psales] to desktop 
# from variable explorer
# or, export through code

ps = pd.DataFrame(psales)
ps.to_csv('ps.csv')

ps = ps.sort_values(by="Sales", ascending=False) # highest, 1st

#12 Quantity 
df.info()
df.Quantity.describe()
'''
count    2121.000000
mean        3.785007
std         2.251620
min         1.000000
25%         2.000000
50%         3.000000
75%         5.000000
max        14.000000
Name: Quantity, dtype: float64

'''
sns.distplot(x=df.Quantity)
sns.boxplot(x = df.Quantity)

#13 Discount 
df.info()
df.Discount.describe()
'''
count    2121.000000
mean        0.173923
std         0.181547
min         0.000000
25%         0.000000
50%         0.200000
75%         0.300000
max         0.700000
Name: Discount, dtype: float64

'''
sns.distplot(x=df.Discount)
sns.boxplot(x = df.Discount)

#14 Profit
df.info()
df.Profit.describe()
'''
count    2121.000000
mean        8.699327
std       136.049246
min     -1862.312400
25%       -12.849000
50%         7.774800
75%        33.726600
max      1013.127000
Name: Profit, dtype: float64

'''
sns.distplot(x=df.Profit)
sns.boxplot(x = df.Profit)

#________eda over, lets export this file for modeling
df.to_csv('ssts.csv')



