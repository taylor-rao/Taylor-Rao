
# coding: utf-8

# In[1]:

#add home or away column
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dat = pd.read_csv('original_game_data.csv')

bat=dat[dat.BatterTeam=='TEX_AGG']


# In[2]:

bat.head()


# In[3]:

bat.iloc[:, 0:20].head(10)


# In[4]:

print(bat.groupby('PitchCall').size())


# In[5]:

print(bat.groupby('Batter').size())


# In[6]:

bat.isnull().sum()


# In[7]:

droplist = ['HangTime','ExitSpeed','Angle','Direction','HitSpinRate','PositionAt110X','PositionAt110Y','PositionAt110Z','Distance','LastTrackedDistance','Bearing', 'ERA', 'BatAvg']  
bat = bat.drop(droplist, axis= 1)


# In[8]:

bat = bat.dropna(how='any')


# In[9]:

y = bat.PitchCall
y = y.replace({'BallCalled': 0, 'StrikeSwinging': 1,'StrikeCalled': 2, 'InPlay' : 3, 'FoulBall' : 4, 'HitByPitch': 5, 'Undefined': 6, 'BallIntentional': 7 })


# In[10]:

myList1 = ['RelSpeed','VertRelAngle', 'InducedVertBreak','HorzRelAngle','SpinRate','SpinAxis','Tilt','RelHeight','RelSide','Extension',
         'VertBreak','HorzBreak','PlateLocHeight','PlateLocSide','ZoneSpeed','VertApprAngle',
         'HorzApprAngle','ZoneTime','pfxx','pfxz','x0','y0','z0','vx0','vy0',
         'vz0','ax0','ay0','az0', 'Inning', 'Week', 
           'PitcherThrows', 'BatterSide', 'TaggedPitchType', 'AutoPitchType', 'Week_Factor', 'Batter', 'BatterSide']
X1 = bat[myList1]


# In[11]:

'BatterSide' in myList1


# In[12]:

list(bat)


# In[13]:

dataDummies = pd.get_dummies(X1)


# In[14]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataDummies, y, random_state=0)


# In[15]:

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[16]:

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# In[17]:

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[18]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
#Accuracy of K-NN classifier on training set: 0.68 w/ batter
#Accuracy of K-NN classifier on test set: 0.52 w/ batter


# In[19]:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


# In[20]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
#Accuracy of GNB classifier on training set: 0.32 w/ batter
#Accuracy of GNB classifier on test set: 0.30 w/ batter


# In[21]:

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
#Accuracy of SVM classifier on training set: 0.60
#Accuracy of SVM classifier on test set: 0.62


# In[22]:

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)


# In[23]:

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))


# In[34]:

myList = []
funlist = list(dataDummies.columns)
for i in funlist:
    if '_' in i:
        tempList = []
        s1 = i.split('_')[0]
        for j in funlist:
            if s1 == j.split('_')[0]:
                tempList.append(j)
        myList.append(tempList)
    else:
        myList.append(i)
        
def removeDuplicates(listofElements):
    
    # Create an empty list to store unique elements
    uniqueList = []
    
    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)
    
    # Return the list of unique elements        
    return uniqueList

myList = removeDuplicates(myList)

del myList[-1]





get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]

def forward(k, method):
    pla = dataDummies.iloc[:,55:71]
    pvar = np.repeat([None], k)
    names = ['none', 'batters']
    
    if method == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        logreg = LogisticRegression()
        
        X_train, X_test, y_train, y_test = train_test_split(pla, y, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)        
        logreg.fit(X_train, y_train)
        player = logreg.score(X_test, y_test)
    
        for j in range(0,k):
            accList = []
            for i in myList:
                data = pd.concat([pla,dataDummies[i]], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0)

                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                logreg.fit(X_train, y_train)
                accList.append(logreg.score(X_test, y_test))
                
            pvar[j] = max(accList)    
            ind = accList.index(max(accList))
            bla = str(myList[ind])
            names.append(bla[:].split('_')[0])
            pla = pd.concat([pla, dataDummies[myList[ind]]] , axis = 1)
            del myList[ind] 
            
    if method == 'DecisionTree':    
       
        from sklearn.tree import DecisionTreeClassifier
        X_train, X_test, y_train, y_test = train_test_split(pla, y, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)        
        clf = DecisionTreeClassifier().fit(X_train, y_train)
        player = clf.score(X_test, y_test)
    
        for j in range(0,k):
            accList = []
            for i in myList:
                data = pd.concat([pla,dataDummies[i]], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0)

                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                clf.fit(X_train, y_train)
                accList.append(clf.score(X_test, y_test))
                
            pvar[j] = max(accList)    
            ind = accList.index(max(accList))
            bla = str(myList[ind])
            names.append(bla[:].split('_')[0])
            pla = pd.concat([pla, dataDummies[myList[ind]]] , axis = 1)
            del myList[ind] 
            
    if method == 'RandomForest':    
       
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators = 100, random_state = 42)  # tree size is reduced for speed
        X_train, X_test, y_train, y_test = train_test_split(pla, y, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)        

        
        rf.fit(X_train, y_train)
        player = rf.score(X_test, y_test)
    
        for j in range(0,k):
            accList = []
            for i in myList:
                data = pd.concat([pla,dataDummies[i]], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0)

                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                rf.fit(X_train, y_train)
                accList.append(rf.score(X_test, y_test))
                
            pvar[j] = max(accList)    
            ind = accList.index(max(accList))
            bla = str(myList[ind])
            names.append(bla[:].split('_')[0])
            pla = pd.concat([pla, dataDummies[myList[ind]]] , axis = 1)
            del myList[ind] 
            
    pvar = np.insert(pvar, 0, player)

    novar = max(bat.groupby('PitchCall').size())/len(bat.PitchCall)
    pvar = np.insert(pvar, 0, novar)


    df = pd.DataFrame({'lab': names, 'val': pvar})
    ax = df.plot.bar(x='lab', y='val', rot=0)
    ax.plot()
    print(pvar)


# In[25]:

forward(10, 'LogisticRegression')


# In[28]:

forward(10, 'DecisionTree')


# In[35]:

forward(10, 'RandomForest')


# In[ ]:

#PlateHeight = 1.084302
#PlateSide = -0.382324
#RelAng = -3.393975
#PitchType = 'Curveball'

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

Players = X1.Batter
Tagged = X1.TaggedPitchType
PlateLocH = X1.PlateLocHeight
PlateLocS = X1.PlateLocSide
RelVertAng = X1.VertRelAngle

data = pd.concat([Tagged, PlateLocH, PlateLocS, RelVertAng, Players], axis =1)

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

PlateHeight = float(input('enter PlateLocHeight: '))
PlateSide = float(input('enter PlateLocSide: '))
RelAng = float(input('enter VertRelAng: '))
PitchType = input('enter TaggedPitchType: ')
print('\n')
#data2 = pd.DataFrame([[PitchType, PlateHeight, PlateSide, RelAnd, dud]])

for i in data.Batter.unique():
    data = pd.concat([Tagged, PlateLocH, PlateLocS, RelVertAng, Players], axis =1)
    data2 = pd.DataFrame([[PitchType, PlateHeight, PlateSide, RelAng, i]], columns = ['TaggedPitchType', 'PlateLocHeight', 'PlateLocSide','VertRelAngle','Batter'])
    #print(i)
    data = data.append(data2)
    #print(data2)
    dummies = pd.get_dummies(data)
    x = dummies.iloc[-1,:]
    dummies = dummies[:-1]
    #print(x)
    #print(dummies.shape)-3.0
    rf.fit(dummies, y)
    rfList = rf.predict_proba([x])
    #print(i, rfList[0][1])
    print(i,':','\n','BallCalled: ', rfList[0][0],'   ','StrikeCalled: ', rfList[0][1],'   ','Inplay: ', rfList[0][2],'   ','FoulBall: ', rfList[0][3],'   ','HitbyPitch: ', rfList[0][4],'   ','Undefined: ', rfList[0][5],'   ','BallIntentional: ', rfList[0][6],'\n')
    
#Ballcalled:0 StrikeSwinging': 1,'StrikeCalled': 2, 'InPlay' : 3, 'FoulBall' : 4, 'HitByPitch': 5, 'Undefined': 6, 'BallIntentional': 7 })


# In[ ]:

data

Players = X1.Batter
Tagged = X1.TaggedPitchType
PlateLocH = X1.PlateLocHeight
PlateLocS = X1.PlateLocSide
RelVertAng = X1.VertRelAngle

data = pd.concat([Tagged, PlateLocH, PlateLocS, RelVertAng, Players, bat.PitchCall], axis =1)
# In[ ]:



