from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Preprocessing import X, y
import joblib

#Split into train and test 20/80
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#scale features
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

#Save the scaler for Branch 2
joblib.dump(scaler, 'scaler.pk1')

