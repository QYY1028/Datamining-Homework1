import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets,ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

data_name = r'D:\研一下学期\数据挖掘\互评作业1\Chicago\building-violations.csv'
txt=open(r'D:\研一下学期\数据挖掘\互评作业1\chicago_reviews_result1.txt','w',encoding = 'utf-8')
problem_1 = 0
def fiveNumber(nums):
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.percentile(nums,25)
    Median=np.median(nums)
    Q3=np.percentile(nums,75)
    
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR #下限值
    upper_limit=Q3+1.5*IQR #上限值
    
    return Minimum,Q1,Median,Q3,Maximum,lower_limit,upper_limit

def set_missing_ages(df):

    age_df = df[['INSPECTION NUMBER', 'PROPERTY GROUP', 'Community Areas']]
    known_Community = age_df.loc[(age_df['Community Areas'].notnull())]
    unknown_Community = age_df[(age_df['Community Areas'].isnull())]
    
    age_df = df[['INSPECTION NUMBER', 'PROPERTY GROUP', 'Census Tracts']]
    known_Census = age_df.loc[(age_df['Census Tracts'].notnull())]
    unknown_Census = age_df[(age_df['Census Tracts'].isnull())]
    
    age_df = df[['INSPECTION NUMBER', 'PROPERTY GROUP', 'Wards']]
    known_Wards = age_df.loc[(age_df['Wards'].notnull())]
    unknown_Wards = age_df[(age_df['Wards'].isnull())]
    
    age_df = df[['INSPECTION NUMBER', 'PROPERTY GROUP', 'Historical Wards 2003-2015']]
    known_Historical = age_df.loc[(age_df['Historical Wards 2003-2015'].notnull())]
    unknown_Historical = age_df[(age_df['Historical Wards 2003-2015'].isnull())]
    
    
    y1 = known_Community.values[:, 0]
    X1 = known_Community.values[:, 1:]
    rfr1 = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr1.fit(X1, y1)
    
    y2 = known_Census.values[:, 0]
    X2 = known_Census.values[:, 1:]
    rfr2 = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr2.fit(X2, y2)
    
    y3 = known_Wards.values[:, 0]
    X3 = known_Wards.values[:, 1:]
    rfr3 = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr3.fit(X3, y3)
    
    y4 = known_Historical.values[:, 0]
    X4 = known_Historical.values[:, 1:]
    rfr4 = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr4.fit(X4, y4)

    predictedAges = rfr1.predict(unknown_Community.values[:, 1:])
    df.loc[ (df['Community Areas'].isnull()), ['Community Areas']] = predictedAges 
    
    predictedAges = rfr2.predict(unknown_Census.values[:, 1:])
    df.loc[ (df['Census Tracts'].isnull()), ['Census Tracts']] = predictedAges 
    
    predictedAges = rfr3.predict(unknown_Wards.values[:, 1:])
    df.loc[ (df['Wards'].isnull()), ['Wards']] = predictedAges 
    
    predictedAges = rfr4.predict(unknown_Historical.values[:, 1:])
    df.loc[ (df['Historical Wards 2003-2015'].isnull()), ['Historical Wards 2003-2015']] = predictedAges 

    return df, rfr

def knn_missing_filled(data, k = 3, dispersed = True):
  att_loss = ['Community Areas', 'Census Tracts', 'Wards', 'Historical Wards 2003-2015']
  data_lost4 = data.copy()
  for att in att_loss:
    print(att)
    x_train = data_lost4[data_lost4[att].notnull()].loc[:,['INSPECTION NUMBER', 'PROPERTY GROUP']]
    y_train = data_lost4[data_lost4[att].notnull()][att]
    test = data_lost4[data_lost4[att].isnull()].loc[:,['INSPECTION NUMBER', 'PROPERTY GROUP']]

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    data_lost4.loc[ (data_lost4[att].isnull()), att ] = clf.predict(test)

  return data_lost4

file = pd.read_csv(data_name)
print(file.columns)
atts = file.columns
num=0
print(file['INSPECTION NUMBER'].isnull())
if problem_1:
  for i in atts[1:]:
    print(i)
    if i!= 'points' and i!= 'price' and i!='description':
        dict = {}
        for j in file.loc[:,i]:
            if j in dict.keys():
                dict[j]+=1
            else:
                dict[j]=1
                num+=1
    
            
        txt.write(i+'\n')
        txt.write(str(dict))
        txt.write('\n')
        print(num)
  txt.close()
# ['INSPECTION NUMBER', 'PROPERTY GROUP', 'Community Areas', 'Census Tracts', 'Wards', 'Historical Wards 2003-2015']
# null_test = file['region_2'][1]
# print(math.isnan(null_test))

att_pool = ['INSPECTION NUMBER', 'PROPERTY GROUP', 'Community Areas', 'Census Tracts', 'Wards', 'Historical Wards 2003-2015']

for att in att_pool:
  num_list = []
  # dict_hist = {}
  nan_count = 0
  for point in file.loc[:,att]:
   if not math.isnan(point):
    num_list.append(point)
   else:
    nan_count+=1
  if att == 'points':
    bins_set = 10
  else:
    bins_set=1000
  point_np = np.array(num_list)
  minn,Q1,mediann,Q3,maxx,lower_limit,upper_limit = fiveNumber(num_list)
  print(att+' five number:')
  print(minn,Q1,mediann,Q3,maxx,lower_limit,upper_limit)
  print('nan count:'+str(nan_count))

# 
#hist
  plt.hist(num_list,edgecolor='k', alpha=0.35,bins=bins_set)
  plt.show()
  # 
#boxplot
  fig = plt.figure(figsize=(8, 6))
  plt.boxplot(num_list, notch=False, sym='o', vert=True)

  # plt.xticks([x+1 for x in range(len(num_list))], ['x1', 'x2', 'x3', 'x4'])
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.show()

#for data lose
#b
  ser = pd.Series(point_np)
  mode_data = ser.mode()
  fix_list = []
  for point in file.loc[:,att]:
   if not math.isnan(point):
    fix_list.append(point)
   else:
    fix_list.append(mode_data)
  fig = plt.figure(figsize=(8, 6))
  plt.boxplot(fix_list, notch=False, sym='o', vert=True)
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.show()
  # 
  

#c
data_c,data_cc = set_missing_ages(file)
data_c.to_csv('/media/sde/qiyayun/data_mining/new-building-violations.csv')
atts=['Community Areas', 'Census Tracts', 'Wards', 'Historical Wards 2003-2015']
nan_count = 0
for att in atts:
  # fig = plt.figure(figsize=(8, 6))
  plt.boxplot(data_c[att], notch=False, sym='o', vert=True)
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.show()
  plt.hist(data_c[att],edgecolor='k', alpha=0.35,bins=10)
  plt.show()

#d
data_d = knn_missing_filled(file)
atts=['Community Areas', 'Census Tracts', 'Wards', 'Historical Wards 2003-2015']
# bin_set=[10,50,]
for att in atts:
  plt.subplot(1,2,1)
  # fig = plt.figure(figsize=(8, 6))
  plt.boxplot(data_d[att], notch=False, sym='o', vert=True)
  plt.title('box plot:'+att)
  plt.xlabel('x')
  plt.subplot(1,2,2)
  plt.hist(data_d[att],edgecolor='k', alpha=0.35,bins=10)
  plt.show()
# plt.show()
