from sklearn.datasets import load_iris, load_digits, load_boston, load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris.DESCR)

dimension_max = iris.data.shape[1]
dimension_tot = iris.data.shape[1]
sample_size = iris.data.shape[0]
nb_of_values =9
forward_computation_mode = False
work_on_transpose = False
supervised_mode = False
sampling_mode = 1
deformed_probability_mode = False

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))
sns.pairplot(iris_df, hue='species')
plt.show()