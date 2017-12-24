import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()


# print boston.keys()
print boston.feature_names
print boston.DESCR
# print boston.data[:5]
# print boston.target[:5]
# print boston.data.shape
# print boston.target.shape

X = boston.data
y = boston.target

# visualise relationship between value and key features
# plt.scatter(y, X[:,5])
# plt.xlabel("rooms")
# plt.ylabel("Prices")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1 )

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# visualise relationship between Actual and predictions
# plt.scatter(y_test, predictions)
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.show()

# r square score
print model.score(X_test, y_test)

# mean squared error
print metrics.mean_squared_error(y_test, predictions)

exit()

