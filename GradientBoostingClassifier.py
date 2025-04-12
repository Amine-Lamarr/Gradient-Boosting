from sklearn.datasets import load_iris 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV , train_test_split
from sklearn.metrics import confusion_matrix , precision_score , classification_report , recall_score , mean_squared_error
import matplotlib.pyplot as plt

data = load_iris()

x = data.data
y = data.target

# splitting data
x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.3 , random_state=23)
# model 
model = GradientBoostingClassifier()
# finding the best hyperparams 
params = {
    'n_estimators' : [100 , 120 , 140 , 160 , 185],
    'learning_rate' : [ 0.01  , 0.05 , 0.1 , 0.8 , 1 ] , 
    'max_depth' : [4 , 6 ,10 , 12 ,14 ,18] , 
    'min_samples_leaf' : [2 ,4 ,6 , 8 , 10 ,12 , 15] ,
    'min_samples_split' : [4 ,8 , 10 , 12 ,15 , 18 ,20] , 
    'subsample' : [0.7 , 0.8 , 0.6 ] , 
    'max_features' : [0.2 , 0.3 , 0.5 , 0.6 , 3 , 2] , 
}
vals = RandomizedSearchCV(model , params , cv=4 , random_state=12)
vals.fit(x_train , y_train)

best_estimators = vals.best_estimator_
best_score = vals.best_score_

print("best estimators : " , best_estimators)
print(f"best score : {best_score*100:.2f}%")

# best estimators :  GradientBoostingClassifier(learning_rate=0.01, max_depth=14, max_features=3, min_samples_leaf=8, min_samples_split=12, n_estimators=185, subsample=0.8)
# best score : 94.34%

# starting our model 
model = best_estimators
model.fit(x_train , y_train)
print("important features : \n" , model.feature_importances_)
# predictions 
predictions = model.predict(x_test)
# score accuracy 
acc_train = model.score(x_train , y_train)
acc_test = model.score(x_test , y_test)

# testing results 
conf = confusion_matrix(y_test , predictions)
precision = precision_score(y_test , predictions , average='macro')
recall = recall_score(y_test , predictions , average='macro')
clfr = classification_report(y_test , predictions)
mse = mean_squared_error(y_test , predictions)

# printing scores 
print(f"accuracy (train): {acc_train*100:.2f}%")
print(f"accuracy (test) : {acc_test*100:.2f}%")
print(f"precision : {precision*100:.2f}%")
print(f"recall score : {recall*100:.2f}%")
print(f"mean squared error : {mse*100:.2f}%")
print(f"classification report : \n {clfr}")

#best score : 94.34% | accuracy (train): 100.00% | accuracy (test) : 97.78% | precision : 97.78% | recall score : 97.44% | mean squared error : 2.22%
# plotting the feature importance
feat_imp = model.feature_importances_
feat_name = data.feature_names
plt.style.use("fivethirtyeight")
plt.bar( feat_name , feat_imp , label ='important features')
plt.xlabel('features')
plt.ylabel("importance")
plt.title("feature importance - GradientBoostingClassifier")
plt.legend()
plt.show()
