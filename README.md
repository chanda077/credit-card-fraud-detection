# credit-card-fraud-detection
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import warnings
warnings.filterwarnings("ignore")


#data selection
print("Data selection")
print()
data = pd.read_csv('creditcard_1.csv')
print(data.head(10))
print()


#Checking missing values
print("checking missing values")
print()
print(data.isnull().sum())
print()

#Data visualization
# number of fraud and valid transactions 
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency");


#Assigning the transaction class "0 = NORMAL  & 1 = FRAUD"
Normal = data[data['Class']==0]
Fraud = data[data['Class']==1]
outlier_fraction = len(Fraud)/float(len(Normal))
print()
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Normal)))


#Training and testing sets
X = data.iloc[:,:-1] 
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)




#RF
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators = 100)  
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
print("--------------------------------------------------------")
print("Random Forest")
print()

"Analysis Report"
print()
print("------Classification Report------")
print(classification_report(rf_prediction,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{(accuracy_score(rf_prediction,y_test)*100)}")
print()

#DT
from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
Result_2=accuracy_score(y_test, y_pred_dt) *100
print("-------------------------------------------------------")
print()
