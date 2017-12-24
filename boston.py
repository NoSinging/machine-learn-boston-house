from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

x = boston.data
y = boston.target

print boston.feature_names
print x
print y

exit()

