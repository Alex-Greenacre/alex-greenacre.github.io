---
layout: post
title:  "Analyzing if violent crime increased during lockdown - Uni Data Analysis Report"
date:   2025-05-29 11:15:00 +0100
categories: report
---

# 1. Introduction 
When looking into domestic crime during lockdown the Economist [1] found that there was a rise of domestic violence during lockdown, this goes against the trend in crime, as shown by a review of crime in London over the same period by [2] who found that crime decreased. Because of this change in trend the report will research violent crime before and during lockdown to see if violent crime increased with domestic violence[1] or followed the trend set by the rest of crime[2].    

The aims of this project is to answer the question that lockdown effected the violent crime rates in the uk, to do this the question will be broken down into three main objectives that will be reviewed these are 
1: Does the data show a increase in crime and if so does the data recorded in 2020
relate to this 
2: Does the data of 2020 variate from that of forecasted data
3: How did lockdown effect crime at a local level

To test both the main hypothesis and a the 3 objectives I analyse and filter the data set provided, before applying it to numerous visitation and analytical techniques to evaluate each objective against the data and concluding if the data shows that both the objectives and wider hypothesis have been proven.  


# 2. Component Selection and Data Pipeline Implementation



```python
# Libaries used 
import warnings 
from datetime import datetime as dt

import geopandas as gpd
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession 
from pyspark.sql import functions as func

from pyspark.ml.stat import Correlation
from pyspark.ml.stat import Summarizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
```


```python
localThreadCount = 8
maxThreadMemory = "4g"
spark = SparkSession.builder.master("local["+str(localThreadCount)+"]")\
.appName("Python Spark SQL")\
.config("spark.driver.maxResultSize",maxThreadMemory)\
.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled","true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled","true")

#check to see if spark started correctly 
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://36ca25585273:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.1.2</code></dd>
      <dt>Master</dt>
        <dd><code>local[8]</code></dd>
      <dt>AppName</dt>
        <dd><code>Python Spark SQL</code></dd>
    </dl>
</div>

    </div>




# 3. Data Extraction and Filtering 
### includes System running, test and diagnostics, 


## 3.1 Extracting the data 
Extracting the dataset from the file and storing it as a pyspark dataframe  


```python
crimeData = spark.read.csv('Data/all_crimes21_hdr.txt',header="true")
```

## 3.2 Filtering Dataset
Once the main file has been extracted three actions are made before storing the refined dataframe
1. The date is formated and only the required columns are selected
2. All only data with violent crimes types are taken
3. The data is ordered by date 


```python
violentCrimeData = crimeData.select(func.date_format('Month','yyyy-MM').alias('Date'),\
                                   crimeData['Crime type'].alias('Crime_Type'),\
                                   crimeData['LSOA name'].alias('Location'),\
                                   crimeData['Longitude'],\
                                   crimeData['Latitude'])\
                            .where(crimeData['Crime type'].isin("Violence and sexual offences","Violent crime")).orderBy("Date")
```


```python
#Check only crime types are shown 
violentCrimeData.groupBy('Crime_Type').count().show()
```

    +--------------------+--------+
    |          Crime_Type|   count|
    +--------------------+--------+
    |       Violent crime| 1673219|
    |Violence and sexu...|11411540|
    +--------------------+--------+
    



```python
#Check date has formatted correctly 
violentCrimeData.columns
```




    ['Date', 'Crime_Type', 'Location', 'Longitude', 'Latitude']



## 3.3 Filtering by objective 
As Differenct sections of the report will require seperate data the violent crime data is then filtered down into smaller dataframes that match the needs of the section
**NOTE** conversion to pandas will only occur during development 


```python
violentCrimeDatesFullWithCount = violentCrimeData.groupBy('Date').count()
violentCrimeDatesFullWithCount = violentCrimeDatesFullWithCount.select(violentCrimeData.Date,\
                                 violentCrimeDatesFullWithCount['count'].alias('Total_Crimes')).orderBy("Date")
#create a pandas version of the variable 
vcCrimeDatesWithCountPandas = violentCrimeDatesFullWithCount.toPandas()
```


```python
violentCrimeLocationData = violentCrimeData.select(violentCrimeData.Location,violentCrimeData.Longitude,\
                                                   violentCrimeData.Latitude,violentCrimeData.Date).orderBy("Date")
#Convert Location dataset from LSOA to Local area district (LAD)
violentCrimeLadData = violentCrimeLocationData.select(violentCrimeData.Longitude,violentCrimeData.Latitude,violentCrimeLocationData.Location,violentCrimeLocationData.Date)
#Remove LSOA Code at the end of each string, keeping the districts name 
violentCrimeLadData =violentCrimeLadData.withColumn("Location", func.expr("substring(Location,0,length(Location)-5)"))  
```

# 4 Design, Development and reasoning 
## 4.1 Has Crime Increased


To look at the increase of crime the data should first be seen from a general overview, to check for any increase the trend, correlation and seasonality of the entire data, should be visualised to check the status of these before any addition statistical analysis can be carried out. 

### 4.1.1 Trend Of Crime - Visualisation
#### Overall Trend 
To view the trend all of the data the data should be displayed as a line graph with the months and year as the x axis and number of crimes committed in that month as the y, this graph should show if there’s any increase in crime as time increases 



```python
#Filter out warnings in notebook to avoid any being displayed later on 
warnings.filterwarnings("ignore")

#Record execution time 
executionStart = dt.now()
#converting pysprak datetime to pandas dt allows for a less cluttered x axis label 
plt.plot(pd.to_datetime(vcCrimeDatesWithCountPandas.Date),vcCrimeDatesWithCountPandas.Total_Crimes,label="Crimes Per Month 2010-2021")
plt.xlabel('Year')
plt.ylabel('Crime rate')
plt.title('Crime rate by year')
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_18_0.png)
    


#### Trend Results 
As shown by the graph as time moves on there is a increase in time this shows that there Is a increase in time one that data 2020 does not change, although a slowdown in this trend is shown from 2018 onwards showing the trend to stabilise and not as positive as 2013-2017
### Correlation
After the general trend is identified the strength of this trend should be checked to see if any change has been made, to do this a scatter graph will be used, this will take the same date and crime amount x and y values as the line graph in trend but unlike the line graph it will show each data entry as is own point, this will help show the strength of the relationship between time and a increase in time as points closer together will show a close relationship where’s points more scattered show a weak relationship.  

Based off the initial results of the correlation graph two subplots showing the correlation before and after 2015 have been added to show the change in correlation strength. 


```python
violentCrimeDatesBefore2015 = vcCrimeDatesWithCountPandas.loc[vcCrimeDatesWithCountPandas.Date <str(2015)]
violentCrimeDatesAfter2015 = vcCrimeDatesWithCountPandas.loc[vcCrimeDatesWithCountPandas.Date >=str(2015)]
plt.subplot(2,1,1)
plt.scatter(pd.to_datetime(vcCrimeDatesWithCountPandas.Date),vcCrimeDatesWithCountPandas.Total_Crimes)
plt.xlabel('Year')
plt.ylabel('Crime rate')
plt.title('Crime rate by year')

plt.subplot(2,2,3)
plt.scatter(pd.to_datetime(violentCrimeDatesBefore2015.Date),violentCrimeDatesBefore2015.Total_Crimes)
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Crime rate')
plt.title('Crime rate 2011-2015')

plt.subplot(2,2,4)
plt.scatter(pd.to_datetime(violentCrimeDatesAfter2015.Date),violentCrimeDatesAfter2015.Total_Crimes)
plt.xticks(rotation=90)
plt.xlabel('Year')
plt.ylabel('Crime rate')
plt.title('Crime rate 2018-2021')

plt.subplots_adjust(wspace=0.5,hspace=0.4)
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_20_0.png)
    


#### Visualisation Results
The results of this graph seem to confirm the trend shown in the line graph, where’s there was a increase in 2013 the strength of this trend seems to become weaker the more time increases.

### Yearly Trends 
Having viewed the data as a whole it should then be viewed on a year to year basis, this will highlight any abnormal behaviour of a certain year as well as show any yearly trends. 
To do this a multi-line graph with each year displayed will be used.



```python
#As Date is YYYY-mm
#Date[:-3] == year
#Date[-2:] == month
for i in range(int(vcCrimeDatesWithCountPandas.Date.min()[:-3])+1,int(vcCrimeDatesWithCountPandas.Date.max()[:-3])+1):
    #finds all dates between the year of i and the next year(i+1)  
    yearlyCrimeMetrics = vcCrimeDatesWithCountPandas.loc[(vcCrimeDatesWithCountPandas.Date >= str(i))\
                                                          & (vcCrimeDatesWithCountPandas.Date < str(i+1))]
    #once the data for the year have been found the month is extracted from the year and set to x axis 
    plt.plot(yearlyCrimeMetrics.Date.str[-2:],yearlyCrimeMetrics.Total_Crimes.values,label = str(i))
#Check last entry to esure data is being formatted properly 
print(yearlyCrimeMetrics)
#Set Legend and show chart 
plt.legend(loc='best',bbox_to_anchor=(0,0,-0.2,1),title="Year")
plt.show()
```

            Date  Total_Crimes
    121  2021-01        141414
    122  2021-02        135101
    123  2021-03        166668
    124  2021-04        153789
    125  2021-05        171614



    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_22_1.png)
    


#### Visualisation Results
The output of the visualisation has shown two clear trends firstly each year follows a seasonal pattern with lows in the winter (February) highs the summer (July/August) secondly the lockdown in 2020 does affect the levels of crime, with a massive drop in the first months of lockdown before a giant spike in the summer before levelling off the tail end of 2020.  


## 4.1.2 Has Crime Increased - Statistical Analysis

Haven identified key trends in ths visualisation two analytical methods can be ran against the data to validate conclusions drawn from them, the first being Correlation this can be ran against the data set to validate that crime tends increased over time before stagnating after 2018, as well as this the statsmodels seasonal decomposition can be ran to verify that a seasonal component is present. 

### Correlation algorithm 


```python
#Converting date to unixtimestamp gives both a usable int value 
#    and will also still keep the trend in time(the time gap between 1 month)
correlationDataFull = violentCrimeDatesFullWithCount.select(func.unix_timestamp(violentCrimeDatesFullWithCount.Date,'yyyy-MM').alias("Date_As_Seconds"),\
                                                    violentCrimeDatesFullWithCount.Total_Crimes,violentCrimeDatesFullWithCount.Date)
correlationData2018 = violentCrimeDatesFullWithCount.select(func.unix_timestamp(violentCrimeDatesFullWithCount.Date,'yyyy-MM').alias("Date_As_Seconds"),\
                                                    violentCrimeDatesFullWithCount.Total_Crimes).where(violentCrimeDatesFullWithCount.Date >= '2018-01')
correlationData2013 = violentCrimeDatesFullWithCount.select(func.unix_timestamp(violentCrimeDatesFullWithCount.Date,'yyyy-MM').alias("Date_As_Seconds"),\
                                                    violentCrimeDatesFullWithCount.Total_Crimes).where((violentCrimeDatesFullWithCount.Date >= '2013-01')\
                                                    &(violentCrimeDatesFullWithCount.Date < '2018-01'))
#Check correlation for each scatter plot range   
strCorrFull = "Full correlation: "+str(correlationDataFull.corr('Date_As_Seconds','Total_Crimes'))
strCorr2013 = "\n 2013-2018: "+str(correlationData2013.corr('Date_As_Seconds','Total_Crimes'))
strCorr2018 = "\nCorrelation between:\n 2018-2021: "+str(correlationData2018.corr('Date_As_Seconds','Total_Crimes'))
print(strCorrFull+strCorr2018+strCorr2013)

#Check corrlation per year 
#Set minimum full year(2011) and max year (2021)
minYear = int(correlationDataFull.select('Date').orderBy('Date').first()[0][:-3])
maxYear = int(correlationDataFull.select('Date').orderBy('Date',ascending=False).first()[0][:-3])
#Loops through each year and checks the correlation for that year  
yearlyCorr = [str(i)+": "+str(correlationDataFull.where((correlationDataFull.Date >= str(i))&\
                                               (correlationDataFull.Date < str(i+1))).corr('Date_As_Seconds','Total_Crimes'))\
 for i in range(minYear+1,maxYear+1)]
print("Correlation by year:"+'\n '.join(yearlyCorr))
```

    Full correlation: 0.9492284414227093
    Correlation between:
     2018-2021: 0.2755811406515931
     2013-2018: 0.9753925336131221
    Correlation by year:2011: -0.31979228769379364
     2012: 0.10953003629286247
     2013: 0.6958649838150631
     2014: 0.8572374826853852
     2015: 0.842590590510554
     2016: 0.8332031922354612
     2017: 0.6879953532172807
     2018: 0.6100821600759098
     2019: -0.0030341072608861073
     2020: 0.24464968469369833
     2021: 0.788440592284602


### Correlation Conclusion 
As shown by the correlation ranges there is a strong positive relationship between time and crime rate during 2013-2018 but after 2018 this becomes a lot weaker with a positive correlation strength being close to a third of what it was in the first range. As well as the time ranges the year to year breakdown shows 2020 did break the trend of the previous 3 years of a gradually weaker relationship, but 2020 still has a half the correlation strength compared to 2017 and 2018.

### Yearly decompisition  


```python
violentCrimeDatesAsIndex = vcCrimeDatesWithCountPandas.set_index(pd.to_datetime(vcCrimeDatesWithCountPandas.Date))
seasonalData = sm.tsa.seasonal_decompose(violentCrimeDatesAsIndex.Total_Crimes,model='additive', extrapolate_trend='freq')
seasonalData.plot()
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_29_0.png)
    


### Yearly decompisition Conclusion 
As shown by the figure produced by the seasonal decomposition the data does follow a seasonal trend with a decrease at the start and a peak in the middle occurring for every year  


### 4.1.3 Has Crime Increased - Conclusion 

A basic overview of the data set does indicate that 2020 did have some of the highest levels of crime with some of the highest levels being of that during lockdown although when scaled out to a yearly level 2020 does not land outside the ongoing trend set by the years before and shows a weak increase in crime compared but still smaller than the postivie correlations of the past 3 years. 


## 4.2 Predicting 2020 Crime Rates 
algorithms can be used to predict levels in 2020 we can then compare these against the data recorded in 2020 to see if there’s is a difference, this should show if crime in 2020 went against what was originally forecasted for that year   


## 4.2.1 Linear Regression  
As explained by [3] linear regression is a model that forecasts data based off a datasets connections between x and y. This can be applied to the crime dataset to predict the data of 2020 based off the previous years data showing if lockdown took crime above levels that would have been predicted.     


```python
trainingData = violentCrimeDatesFullWithCount.select(func.unix_timestamp(violentCrimeDatesFullWithCount.Date,'yyyy-MM').alias("Date_As_Seconds"),\
                                                    violentCrimeDatesFullWithCount.Total_Crimes).where(violentCrimeDatesFullWithCount.Date <= '2019-12')
testData = violentCrimeDatesFullWithCount.select(violentCrimeDatesFullWithCount.Date,func.unix_timestamp(violentCrimeDatesFullWithCount.Date,'yyyy-MM').alias("Date_As_Seconds"),\
                                                    violentCrimeDatesFullWithCount.Total_Crimes).where(violentCrimeDatesFullWithCount.Date >= '2020-01')
testData.columns
```




    ['Date', 'Date_As_Seconds', 'Total_Crimes']




```python
outputTarget="dateVectors"
vectorAsmb = VectorAssembler(inputCols=['Date_As_Seconds'],outputCol=outputTarget)
vectors = vectorAsmb.transform(trainingData)
vectors.head()
```




    Row(Date_As_Seconds=1291161600, Total_Crimes=57580, dateVectors=DenseVector([1291161600.0]))




```python
lr = LinearRegression(featuresCol="dateVectors",labelCol="Total_Crimes")
model = lr.fit(vectors)
testVectors = vectorAsmb.transform(testData)
prediction = model.transform(testVectors)
prediction.show()
```

    +-------+---------------+------------+-------------+------------------+
    |   Date|Date_As_Seconds|Total_Crimes|  dateVectors|        prediction|
    +-------+---------------+------------+-------------+------------------+
    |2020-01|     1577836800|      157274|[1.5778368E9]|158685.66833094019|
    |2020-02|     1580515200|      151260|[1.5805152E9]| 159848.2277532809|
    |2020-03|     1583020800|      149773|[1.5830208E9]|160935.78334192222|
    |2020-04|     1585699200|      127901|[1.5856992E9]|162098.34276426292|
    |2020-05|     1588291200|      145837|[1.5882912E9]|163223.40026975388|
    |2020-06|     1590969600|      158701|[1.5909696E9]| 164385.9596920946|
    |2020-07|     1593561600|      175562|[1.5935616E9]|165511.01719758555|
    |2020-08|     1596240000|      174980|  [1.59624E9]|166673.57661992626|
    |2020-09|     1598918400|      164204|[1.5989184E9]|167836.13604226697|
    |2020-10|     1601510400|      158725|[1.6015104E9]|168961.19354775804|
    |2020-11|     1604188800|      151251|[1.6041888E9]|170123.75297009875|
    |2020-12|     1606780800|      148517|[1.6067808E9]| 171248.8104755897|
    |2021-01|     1609459200|      141414|[1.6094592E9]|172411.36989793042|
    |2021-02|     1612137600|      135101|[1.6121376E9]|173573.92932027113|
    |2021-03|     1614556800|      166668|[1.6145568E9]| 174623.9829920627|
    |2021-04|     1617235200|      153789|[1.6172352E9]| 175786.5424144034|
    |2021-05|     1619827200|      171614|[1.6198272E9]|176911.59991989448|
    +-------+---------------+------------+-------------+------------------+
    


A line chart is best to show the results of linear regression as it can show both the line that the regression algorithm has outputted and how close it comes to represent the actual line 


```python
#warnings.filterwarnings("ignore")

predictionAsPandas = prediction.toPandas()
plt.plot(vcCrimeDatesWithCountPandas.Date,vcCrimeDatesWithCountPandas.Total_Crimes)
plt.plot(predictionAsPandas.Date,predictionAsPandas.prediction)
plt.show()
violentCrimesAsPandas=""
predictionAsPandas=""
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_39_0.png)
    


As shown by the line graph the crime rate in 2020 mainly stays bellow the predicted line indicating that lockdown may not have increased crime rates 


```python
modelEvaluator = RegressionEvaluator(predictionCol="prediction",labelCol="Total_Crimes",metricName="rmse")
accuracy = modelEvaluator.evaluate(prediction)
accuracy
```




    18600.421357608644



## 4.2.2 SARIMAX

As described by [4] SARIMAX is a popular technique that can be used against time series data with seasonality to predict future results. As well as predicting seasonality both [4] and [5] state that SARIMAX can account for external variables meaning that the data may not need to be stationary for accurate results.    
Like linear regression SARIMAX can be used forecast data for 2020 and compare against actual trend but as SARIMAX focus’s is more on the seasonality the model can explore if the spikes and dips in crime rate is linked with seasonality or if it could be a factor of lockdown 

Before any SARIMAX model can be ran the data should be checked for stationarity using the dicky fuller test then if the data is not stationary detrend and de-seasonality should be carried out to make the data stationary. When making the data stationary consideration will be made towards the SARIMAX’s account for external variables [4][5] and a comparison against stationary and non-stationary data will be made.      


```python
def dickyFullerTest(dataFrame):
    testResult = adfuller(dataFrame,autolag='AIC')
    return testResult 

def getBestScore(pdq,seaonalPdq,trainingData):
    bestScore=None
    for x in pdq:
        for seasonalParams in seasonalPdq:
            model =sm.tsa.statespace.SARIMAX(trainingData.values,order=x,seasonal_order=seasonalParams)
            result = model.fit()
            if bestScore == None:
                bestScore = [x,seasonalParams,result.aic]  
            elif result.aic < bestScore[2]:
                bestScore = [x,seasonalParams,result.aic]
    return bestScore
```


```python
violentCrimeDatesStartYearly = violentCrimeDatesAsIndex['2011-01-01':'2020-12-02']
violentCrimeDatesStartYearly
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Total_Crimes</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-01</th>
      <td>2011-01</td>
      <td>59823</td>
    </tr>
    <tr>
      <th>2011-02-01</th>
      <td>2011-02</td>
      <td>56691</td>
    </tr>
    <tr>
      <th>2011-03-01</th>
      <td>2011-03</td>
      <td>62326</td>
    </tr>
    <tr>
      <th>2011-04-01</th>
      <td>2011-04</td>
      <td>64299</td>
    </tr>
    <tr>
      <th>2011-05-01</th>
      <td>2011-05</td>
      <td>63633</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-08-01</th>
      <td>2020-08</td>
      <td>174980</td>
    </tr>
    <tr>
      <th>2020-09-01</th>
      <td>2020-09</td>
      <td>164204</td>
    </tr>
    <tr>
      <th>2020-10-01</th>
      <td>2020-10</td>
      <td>158725</td>
    </tr>
    <tr>
      <th>2020-11-01</th>
      <td>2020-11</td>
      <td>151251</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>2020-12</td>
      <td>148517</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 2 columns</p>
</div>




```python
dickyFullerTest(violentCrimeDatesAsIndex.Total_Crimes)
```




    (-0.30470086187703066,
     0.924835905735687,
     12,
     113,
     {'1%': -3.489589552580676,
      '5%': -2.887477210140433,
      '10%': -2.580604145195395},
     2269.8384976951106)




```python
#Remove trend
violentCrimeDatesDetrend = (violentCrimeDatesStartYearly.Total_Crimes - violentCrimeDatesStartYearly.Total_Crimes.rolling(window=12).mean())\
                           /violentCrimeDatesStartYearly.Total_Crimes.rolling(window=12).std()
dickyFullerTest(violentCrimeDatesDetrend.dropna())
```




    (-1.5366685232571795,
     0.5153003034927373,
     13,
     95,
     {'1%': -3.5011373281819504,
      '5%': -2.8924800524857854,
      '10%': -2.5832749307479226},
     175.49707848818898)




```python
#Test with differencing 
violentCrimeDatesDifferencing = violentCrimeDatesAsIndex.Total_Crimes - violentCrimeDatesAsIndex.Total_Crimes.shift(12)                      
violentCrimeDatesDifferencing
#violentCrimeDatesDetrend
dickyFullerTest(violentCrimeDatesDifferencing.dropna())
```




    (-2.017521516215173,
     0.2789437595215539,
     12,
     101,
     {'1%': -3.4968181663902103,
      '5%': -2.8906107514600103,
      '10%': -2.5822770483285953},
     2002.2061781566756)




```python
#Test with detrend and differencing 
violentCrimeDatesDetrendAndDifferencing = violentCrimeDatesDetrend - violentCrimeDatesDetrend.shift(12)
dickyFullerTest(violentCrimeDatesDetrendAndDifferencing.dropna())

```




    (-4.11573970286673,
     0.0009121036726545564,
     0,
     96,
     {'1%': -3.5003788874873405,
      '5%': -2.8921519665075235,
      '10%': -2.5830997960069446},
     154.00571189703004)




```python
stationaryTrainingData = violentCrimeDatesDetrendAndDifferencing[:'2019-12-02']
stationaryTrainingData = stationaryTrainingData.dropna()
pdqVals = range(0,2)

#Get seasonal values via a grid search 
seasonalPeriod = 12 
pdq = list(itertools.product(pdqVals,pdqVals,pdqVals))
seasonalPdq =[(x[0],x[1],x[2],seasonalPeriod) for x in pdq]

#Search for the best PDQs values 
stationaryBestScore = getBestScore(pdq,seasonalPdq,stationaryTrainingData)

#violentCrimeDatesDetrend
detrendTrainingData = violentCrimeDatesDetrend[:'2019-12-02']
detrendTrainingData = detrendTrainingData.dropna()
bestScoreDetrend = getBestScore(pdq,seasonalPdq,detrendTrainingData)

print("Stationary order values: "+str(stationaryBestScore[:2])+"\nDetrend order values: "+str(bestScoreDetrend[:2]))
print("AIC Score: \n Stationary: "+str(stationaryBestScore[-1])+"\n Detrended: "+str(bestScoreDetrend[-1]))

```

    Stationary order values: [(1, 1, 1), (0, 0, 1, 12)]
    Detrend order values: [(1, 1, 1), (0, 1, 1, 12)]
    AIC Score: 
     Stationary: 104.99799781334823
     Detrended: 104.99799484524861


### Evaluation results 
Having ran both evaluations it is shown that the non-stationary detrended data should be used over the completely stationary data for the SARIMAX model, this is because not only does it give a very slightly better aic score but keeping the seasonality should also allow for visualising the previous and predicted seasonality against the actual 2020 results  


```python
#Fit the model with the best PDQs values 
order,seasonal_order = bestScoreDetrend[0],bestScoreDetrend[1]
model =sm.tsa.statespace.SARIMAX(detrendTrainingData.values,order=order,seasonal_order=seasonal_order)
result = model.fit()
prediction= result.get_forecast(steps=seasonalPeriod)

#Display the visualisation of the model 
plt.plot(violentCrimeDatesDetrend['2010-12-02':'2020-12-01'],label="actual")
plt.plot(violentCrimeDatesDetrend['2019-12-02':'2020-12-01'].index,prediction.predicted_mean,label="predicted")
plt.legend(loc='best',bbox_to_anchor=(0,0,-0.2,1),title="Year")
plt.title("SARIMAX Prediction Results")
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_51_0.png)
    



```python
#check result of the fitted model with the one produced in best score 
print ("Expected: "+str(bestScoreDetrend[-1])+"   Actual: "+str(result.aic))
```

    Expected: 104.99799484524861   Actual: 104.99799484524861


### SARIMAX conclusion 
Having ran the results it is clear that lockdown had some sort of effect against seasonality with the predicted model not matching both the peaks and dips predicted meaning that the spike in violent crime rate was not linked to seasonality and could be a cause of lockdown 


# 4.3 How Local Area Are Effected 

Viewing the data from a district level will give a better idea on how each area was effected by the lockdown, this approach could be useful to see the impact of violent crime at a local level between 2020 and the previous year 

To approach the data locally I will explore 2 main areas 
A visualisation of England as a Choreograph showing the different districts crime level before and during lockdown 
Using k-means to show any changes in the crime hotspots of the country 



```python
mapOfEngland = gpd.read_file("Local_Authority_Districts_May_2022_UK_BFE_V3_2022_2232821895412657640.geojson")#[6] -reference to geoJSON download 
#Check file has been imported successfuly 
mapOfEngland.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OBJECTID</th>
      <th>LAD22CD</th>
      <th>LAD22NM</th>
      <th>BNG_E</th>
      <th>BNG_N</th>
      <th>LONG</th>
      <th>LAT</th>
      <th>GlobalID</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>E06000001</td>
      <td>Hartlepool</td>
      <td>447160</td>
      <td>531474</td>
      <td>-1.27018</td>
      <td>54.67614</td>
      <td>2efc9848-300e-4ef3-a36e-58d6856b9817</td>
      <td>POLYGON ((447213.899 537036.104, 447228.798 53...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>E06000002</td>
      <td>Middlesbrough</td>
      <td>451141</td>
      <td>516887</td>
      <td>-1.21099</td>
      <td>54.54467</td>
      <td>6d66b015-1f67-40f6-b239-15911fa03834</td>
      <td>POLYGON ((448489.897 522071.798, 448592.597 52...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>E06000003</td>
      <td>Redcar and Cleveland</td>
      <td>464361</td>
      <td>519597</td>
      <td>-1.00608</td>
      <td>54.56752</td>
      <td>a5a6513f-916e-4769-bed2-cd019d18719a</td>
      <td>POLYGON ((455525.931 528406.654, 455724.632 52...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>E06000004</td>
      <td>Stockton-on-Tees</td>
      <td>444940</td>
      <td>518183</td>
      <td>-1.30664</td>
      <td>54.55691</td>
      <td>14e8450b-7e7c-479a-a335-095ac2d9a701</td>
      <td>POLYGON ((444157.002 527956.304, 444165.898 52...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>E06000005</td>
      <td>Darlington</td>
      <td>428029</td>
      <td>515648</td>
      <td>-1.56835</td>
      <td>54.53534</td>
      <td>2f212ecf-daf5-4171-b9c6-825c0d33e5af</td>
      <td>POLYGON ((423496.602 524724.299, 423497.204 52...</td>
    </tr>
  </tbody>
</table>
</div>



## 4.3.1 Local Area Heatmap 
To see the effects of the lockdown at district level a heatmap map can be used to see the rate of crimes per region, this can then be compared against the previous year to see if there is any major changes. The confirm any features identified in the graph  a comparision table will also be produced analysing the difference in crime between both years   


```python
#Prepare Data for 2019 
violentCrimeLadData2019 = violentCrimeLadData.where((violentCrimeLadData.Date >='2019-01')&\
                                                          (violentCrimeLadData.Date <='2019-12')).groupBy('Location')\
                                                          .count()
#Once group by has occured rename columns to allow for merge with map
violentCrimeLadData2019 = violentCrimeLadData2019.select(violentCrimeLadData2019.Location.alias('LAD22NM'),\
                                                           violentCrimeLadData2019['count'].alias('total_crimes')).toPandas()
mapOfEngland2019 = mapOfEngland.merge(violentCrimeLadData2019,on="LAD22NM",how='inner')

#Prepare Data for 2020
violentCrimeLadData2020 = violentCrimeLadData.where((violentCrimeLadData.Date >='2020-01')&\
                                                          (violentCrimeLadData.Date <='2020-12')).groupBy('Location').count()
violentCrimeLadData2020 = violentCrimeLadData2020.select(violentCrimeLadData2020.Location.alias('LAD22NM'),\
                                                           violentCrimeLadData2020['count'].alias('total_crimes')).toPandas()
mapOfEngland2020 = mapOfEngland.merge(violentCrimeLadData2020,on="LAD22NM",how='inner')
#Display Heat Map
fig, ax = plt.subplots(figsize=(28,16),nrows=1,ncols=2)
mapOfEngland2019.plot(ax=ax[0],column="total_crimes",legend=True,cmap="OrRd",figsize=(15,10),missing_kwds={"color":"blue"})
mapOfEngland2020.plot(ax=ax[1],column="total_crimes",legend=True,cmap="OrRd",missing_kwds={"color":"blue"})
plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_57_0.png)
    



```python
#Evaluate 
#Check Data for 2019 and compare against 2020 
highCrimeRateComp = mapOfEngland2019.nlargest(10,'total_crimes')
highCrimeRateComp = highCrimeRateComp.rename(columns={'total_crimes':'total_crimes_2019'})
highCrimeRateComp = pd.merge(highCrimeRateComp,mapOfEngland2020[['LAD22NM','total_crimes']],on='LAD22NM',how='left')
highCrimeRateComp = highCrimeRateComp.rename(columns={'total_crimes':'total_crimes_2020'})
highCrimeRateComp['change_in_crime_rate'] = highCrimeRateComp['total_crimes_2020'] - highCrimeRateComp['total_crimes_2019']
print(highCrimeRateComp[['LAD22NM','total_crimes_2019','total_crimes_2020','change_in_crime_rate']])
```

             LAD22NM  total_crimes_2019  total_crimes_2020  change_in_crime_rate
    0     Birmingham              42776              54306                 11530
    1          Leeds              38943              37209                 -1734
    2       Bradford              29851              30354                   503
    3  County Durham              21255              22356                  1101
    4      Liverpool              20626              21774                  1148
    5       Kirklees              18805              18569                  -236
    6      Sheffield              17838              18066                   228
    7      Wakefield              16473              16154                  -319
    8      Leicester              15055              16637                  1582
    9         Medway              14826              14773                   -53


### Heatmap conclusion
The graph seems to confirm that there was some decrease in areas such Leeds/Bradford with both districts dropping a colour grade to a less server colour, but Durham County also dropped a level dispite a increase. This may be due to Birmingham increased crime changing the scaling the colour grade and moving others down a level. The comparison table confirms this as Birmingham increases the most by any of the 2019 hotspots where all others have marginal gains. 

Another aspect presented by the comparison table is the increase of 10.5% in Leicester this could show an increase in lockdown as this area spent over 12 months in lockdown [7] and when compared to the other 9 areas with high crime rates in 2019, only Birmingham increasing crime at a larger percentage with 26.9% over the year before       


## 4.3.2 Hotspotting with k-means
To help show the impact of crime in local areas a k-means clustering algorithm can be used.   

K-means described by Kumar, et al [8] a cluster based algorithm that groups data based on the average distance to a clusters centre. This means that data will be grouped to get the equal amount of data in each label, which in turn will show areas of high crime as small clusters where sparse amounts of data are shown over a larger distance. 
As shown by [9] k-means can be applied crime datasets to show crime hotspots.
Applying this technique to the crime data of 2019 and again in 2020 should identify the spread of crime, it’s hotspots and if there has been any change during lockdown. 



```python
def RunKMeansModel(longAndLatDF,amountOfClusters):
    newModel =None
    #create a new model with the set amount of clusters and a set random sequence 
    newModel = KMeans(n_clusters=amountOfClusters,random_state=1)
    newModel.fit(np.array(longAndLatDF))
    return newModel
def ClusterCentersDF(inputModel):
    clusterResults = pd.DataFrame({
        "Long": [x[0]for x in inputModel.cluster_centers_],
        "Lat": [x[1]for x in inputModel.cluster_centers_]
    })
    return gpd.GeoDataFrame(inputModel.cluster_centers_, geometry=gpd.points_from_xy(clusterResults.Long,clusterResults.Lat),\
                            crs="EPSG:4326")
    
```


```python
clusterCount = 10
vcLongAndLatData2019 = violentCrimeLadData.select(violentCrimeLadData.Longitude,violentCrimeLadData.Latitude).where((violentCrimeLadData.Date >='2019-01')&\
                                                          (violentCrimeLadData.Date <='2019-12')).toPandas().dropna()
vcLongAndLatData2020 = violentCrimeLadData.select(violentCrimeLadData.Longitude,violentCrimeLadData.Latitude).where((violentCrimeLadData.Date >='2020-01')&\
                                                          (violentCrimeLadData.Date <='2020-12')).toPandas().dropna()
```


```python
#Check the maps features and currently set crs 
mapOfEngland.crs
```




    <Projected CRS: EPSG:27700>
    Name: OSGB36 / British National Grid
    Axis Info [cartesian]:
    - E[east]: Easting (metre)
    - N[north]: Northing (metre)
    Area of Use:
    - name: United Kingdom (UK) - offshore to boundary of UKCS within 49°45'N to 61°N and 9°W to 2°E; onshore Great Britain (England, Wales and Scotland). Isle of Man onshore.
    - bounds: (-9.01, 49.75, 2.01, 61.01)
    Coordinate Operation:
    - name: British National Grid
    - method: Transverse Mercator
    Datum: Ordnance Survey of Great Britain 1936
    - Ellipsoid: Airy 1830
    - Prime Meridian: Greenwich




```python
#Switching to world based crs to allow longitude and latitude data to be plotted  
FormattedMap = mapOfEngland.to_crs('EPSG:4326')
```


```python
modelResults2019 = RunKMeansModel(vcLongAndLatData2019,clusterCount)
#Check cluster centers are coordinates 
modelResults2019.cluster_centers_
```




    array([[-1.92842485, 52.50542801],
           [ 0.88948238, 51.72058316],
           [-2.88193616, 53.54246044],
           [-3.48637286, 51.26105083],
           [-0.1716334 , 51.50621949],
           [-1.44881434, 54.79543227],
           [-6.29230942, 54.64003995],
           [-1.66204981, 53.63422078],
           [-0.71199778, 53.00507672],
           [-1.46661127, 51.08553147]])




```python
mapOfClusteres2019 = ClusterCentersDF(modelResults2019)
#Check clusters have been created as a dataset 
mapOfClusteres2019.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.928425</td>
      <td>52.505428</td>
      <td>POINT (-1.92842 52.50543)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.889482</td>
      <td>51.720583</td>
      <td>POINT (0.88948 51.72058)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.881936</td>
      <td>53.542460</td>
      <td>POINT (-2.88194 53.54246)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-3.486373</td>
      <td>51.261051</td>
      <td>POINT (-3.48637 51.26105)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.171633</td>
      <td>51.506219</td>
      <td>POINT (-0.17163 51.50622)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#As kmeans creates labels in order of the data inputted the labels can be assigned to a new column next to the inputed values 
vcLongAndLatData2019['Labels'] = modelResults2019.labels_
vcLabelMap2019 = gpd.GeoDataFrame(vcLongAndLatData2019.Labels,\
                                  geometry=gpd.points_from_xy(vcLongAndLatData2019.Longitude,vcLongAndLatData2019.Latitude),\
                                  crs="EPSG:4326")
#Check label dataset has been created 
vcLabelMap2019.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Labels</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>POINT (-2.50938 51.40959)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>POINT (-2.51157 51.41490)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>POINT (-2.50913 51.41614)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>POINT (-2.50132 51.41757)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>POINT (-2.49777 51.42023)</td>
    </tr>
  </tbody>
</table>
</div>




```python
modelResults2020 = RunKMeansModel(vcLongAndLatData2020,clusterCount)
#Get cluster centers of 2020 
mapOfClusteres2020 = ClusterCentersDF(modelResults2020)

#Get cluster labels and assign them back to the coordinates 
vcLongAndLatData2020['Labels'] = modelResults2020.labels_
vcLabelMap2020 = gpd.GeoDataFrame(vcLongAndLatData2020.Labels,\
                                  geometry=gpd.points_from_xy(vcLongAndLatData2020.Longitude,vcLongAndLatData2020.Latitude),\
                                  crs="EPSG:4326")
vcLabelMap2020.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Labels</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>POINT (-2.51193 51.40944)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>POINT (-2.50913 51.41614)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>POINT (-2.50913 51.41614)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>POINT (-2.50938 51.40959)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>POINT (-2.49992 51.41737)</td>
    </tr>
  </tbody>
</table>
</div>



### Visualisation Output 
The output of k-means algorithm will be shown on two plots, both will show a plot for 2019 and a plot for 2020 and use a map of the uk as to display the results. The first graph will show the cluster centre positions and the second plot will show the classification boundaries. This will firstly help show if any major change has been made to the classifications before showing any change in the classification boundaries  



```python
fig, ax = plt.subplots(figsize=(28,16),nrows=1,ncols=2)
#2019
#colour way of lighblue with black borders helps define districts 
#       whilst also keeping a good colour contrast for center points  
clusterMap = FormattedMap.plot(ax=ax[0],color="lightblue",edgecolor="black")
mapOfClusteres2019.plot(ax=clusterMap,color="Red")
#2020
clusterMap = FormattedMap.plot(ax=ax[1],color="lightblue",edgecolor="black")
mapOfClusteres2020.plot(ax=clusterMap,color="Red")
ax[0].set_title("2019")
ax[1].set_title("2020")

plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_70_0.png)
    


### Kmeans Cluster Centre Outcome 
as shown by the graph cluster centers have not moved between 2019 and 2020, this shows that no major change in the spread of crimes has occurred because of lockdown.



```python
fig, ax = plt.subplots(figsize=(28,16),nrows=1,ncols=2)
#2019
labelMap2019 = FormattedMap.plot(ax=ax[0],color="white",edgecolor="black")
vcLabelMap2019.plot(ax=labelMap2019,column='Labels',cmap='plasma',legend=True,figsize=(15,20))
#2020
labelMap2020 = FormattedMap.plot(ax=ax[1],color="white",edgecolor="black")
vcLabelMap2020.plot(ax=labelMap2020,column='Labels',cmap='plasma',legend=True,figsize=(15,20))

plt.show()
```


    
![png](/assets/report_assets/2025-06-18-Violent-Crime-Rate-Assignment/output_72_0.png)
    


### Kmeans Classification Outcome 
The classification graphs show that there has been some but no major change between years. There’s a sizable change in the northern part of England around north east England and Yorkshire with the label expanding from county Durham further south, indicating that there has been a decrease in the north east between years dispite the increase of crime in county Durham. 


```python
#Evaluation of each kmean model 
#2019
#Distance from cluster centre 
Inertia2019 = str(modelResults2019.inertia_)
#2020
Inertia2020 = str(modelResults2020.inertia_)
print("Inertia Score:\n   2019: "+Inertia2019+"   2020: "+Inertia2020)
#Calcuate total execution time 
executionEnd = dt.now()
print("Total Time To Run: "+str(executionEnd-executionStart))
```

    Inertia Score:
       2019: 530868.5306627296   2020: 529383.0118123436
    Total Time To Run: 0:11:02.580088



# 5 Detailed Analysis and consideration of the appropriateness of the solution for the initial problem



Applying k-means to this scenario seemed like the most appropriate technique to monitor the changes of data as a local level this can backed up by [10] who describes k-means as being a common machine learning technique when working with crime datasets. Although there are clear advantages to applying this method, one area that may limit the k-means algorithm is the scales in which it was applied, this is due to other literature [8][9] applying k-means against a city or district, where’s the application in this project was across the country, the main effect of this can be seen when trying to evaluate the project, as the inertia evaluation is based off distances the score produced is very high due to the scale of the values entered.

Linear regression also suffers a limitation with scaling with non-normalised data can lead to a issue due to the scale of the dataset, although predicted values will be accurate to the scale, evaluation methods will also reflect the scale of the dataset, this effects the evaluation of the dataset as the metrics will be scaled to high. This has a clear disadvantage a unevaluated model means that there is a risk that the model itself does not provide a accurate result 


# 6 Evaluation and Conclusion  



Having explored a number of aspect of the hypothesis that violent crime increased during lockdown.There is evidence that would support the main hypothesis, during 2020 violent crime did shoot up to peak numbers once lockdowns initially began to ease indicating a relationship between increased crime and the lockdowns as well as this 2020 did reverse the decline in correlation between time and crime rate for the first time in three years, there is also evidence provided in the SARIMAX forecasting which showed that 2020 deviated from predicted seasonality. As well as SARIMAX forecasting looking at Leicester lengthy lockdown period[7] and how crime rates increased during this time does provide evidence that lockdown could be a factor for this increase.       
Although this evidence points towards violent crime increased there is a strong argument to be made against the supporting evidence firstly although 2020 did revert the annually decrease in correlation its change was only to a weak relationship and did not deviate to far from the correlation set the prior year, as well as this forecasts by linear regression showed that the 2020 still fell bellow expected levels despite the spike in crime rates in the summer. The results of research against local areas can also back up this argument, this is because no clear link was found between lockdowns and crime rates both in the k-means algorithm. 

In conclusion there is enough evidence to support that violent crime increased during lockdown. Although given both the evidence supporting and not supporting this, the impact of lockdown is only minimal. 


# 7 References and Citation 
[1] The Economist."Domestic Violence Has Increased during Coronavirus Lockdowns; Daily Chart." *The Economist* Accessed: Jan. 16, 2024.[Online]. Available: https://www.economist.com/graphic-detail/2020/04/22/domestic-violence-has-increased-during-coronavirus-lockdowns  

[2]S. Sharmin, F. I. Alam, A. Das and R. Uddin, "An Investigation into Crime Forecast Using Auto ARIMA and Stacked LSTM," *2022 International Conference on Innovations in Science, Engineering and Technology (ICISET),* Chittagong, Bangladesh, 2022, pp. 415-420, doi: 10.1109/ICISET54810.2022.9775862.

[3] S. Khatun, K. Banoth, A. Dilli, S. Kakarlapudi, S. V. Karrola and G. C. Babu, "Machine Learning based Advanced Crime Prediction and Analysis," *2023 International Conference on Sustainable Computing and Data Communication Systems (ICSCDS),* Erode, India, 2023, pp. 90-96, doi: 10.1109/ICSCDS56580.2023.10104655.

[4]F. Tahseen Mohammad and S. Krupasindhu Panigrahi, "Forecasting Crude Oil Price Using SARIMAX Machine Learning Approach," *2023 International Conference on Sustainable Islamic Business and Finance (SIBF),* Bahrain, 2023, pp. 131-135, doi: 10.1109/SIBF60067.2023.10379964.

[5] B. Chatuanramtharnghaka, S. Deb and K. R. Singh, "Short - Term Load Forecasting for IEEE 33 Bus Test System using SARIMAX," *2023 IEEE 2nd International Conference on Industrial Electronics: Developments & Applications (ICIDeA),* Imphal, India, 2023, pp. 275-280, doi: 10.1109/ICIDeA59866.2023.10295066.

[6] Office for National Statistics."Local Authority Districts (May 2022) UK BFE V3 | Local Authority Districts (May 2022) UK BFE V3 | Open Geography Portal". *Office for National Statistics.* Accessed: Jan. 10, 2024. [Online]. Avalible: https://geoportal.statistics.gov.uk/datasets/196d1a072aaa4882a50be333679d4f63/explore?location=55.493737%2C-3.800226%2C6.16

[7]Leicester City Council. "Beyond the lockdowns: Lessons learned from Leicester's COVID story - Healthy Places" *Leicester City Council* Accessed: Jan. 16, 2024.[Online]. Available: https://www.leicester.gov.uk/content/beyond-the-lockdowns-lessons-learned-from-leicester-s-covid-story/healthy-places/

[8] A. V. Kumar, S. Chitumadugula and V. T. Rayalacheruvu, "Crime Data Analysis using Big Data Analytics and Visualization using Tableau," in 2022 *6th International Conference on Electronics, Communication and Aerospace Technology, Coimbatore,* India, 2022, pp. 627-632, doi: 10.1109/ICECA55336.2022.10009119.

[9] L. S. Thota, M. Alalyan, A. -O. A. Khalid, F. Fathima, S. B. Changalasetty and M. Shiblee, "Cluster based zoning of crime info," in 2017 *2nd International Conference on Anti-Cyber Crimes (ICACC)*, Abha, Saudi Arabia, 2017, pp. 87-92, doi: 10.1109/Anti-Cybercrime.2017.7905269.

[10] B. Kaur, L. Ahuja and V. Kumar, "Crime Against Women: Analysis and Prediction Using Data Mining Techniques," in *2019 International Conference on Machine Learning, Big Data, Cloud and Parallel Computing (COMITCon),* Faridabad, India, 2019, pp. 194-196, doi: 10.1109/COMITCon.2019.8862195.
