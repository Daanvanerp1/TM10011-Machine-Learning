#pip install pandas numpy seaborn scipy

import pandas
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt

#Load the dataset
data = pandas.read_csv("C:\\Users\\daanv\\Machine Learning\\datasets.csv")
#Display the first few rows of the dataset as Dataframe
df = pandas.DataFrame(data)
print(df.head())

#Print nr of datasets (based on the 'dataset' column of the csv file)
print("Number of datasets:", df['dataset'].nunique())

#print names of datasets
print("Names of datasets:", df['dataset'].unique())

#print statistics per dataset (count, mean, variance, std dev) use groupby
print("Statistics per dataset:")
print(df.groupby('dataset').agg(['count', 'mean', 'var', 'std']))

#Creaate violin plots of x-coordinates per dataset, next to each other
plt.figure()
seaborn.violinplot(x='dataset', y='x', data=df)

#Do the same for the y-coordinates
plt.figure()
seaborn.violinplot(x='dataset', y='y', data=df)

#Determine and print correlation between x and y for each dataset
print("Correlation between x and y for each dataset:")
print(df.groupby('dataset').apply(lambda x: scipy.stats.pearsonr(x['x'], x['y'])))

#Determine and print covariance matrix for each dataset
print("Covariance matrix for each dataset:")
print(df.groupby('dataset').apply(lambda x: numpy.cov(x['x'], x['y'])))

#Determine linear regression between x and y for each dataset, and print slope,intercept and r-value for each dataset (hint use scipy.stats.linregress)
print("Linear regression between x and y for each dataset:")
print(df.groupby('dataset').apply(lambda x: scipy.stats.linregress(x['x'], x['y'])))

#Create scatterplots for all datasets (hint: use FacetGrid and map_dataframe)
g = seaborn.FacetGrid(df, col='dataset')
g.map_dataframe(seaborn.scatterplot, x='x', y='y')

#Create scatterplots including the regression line for all datasets (hint: use Implot)
g = seaborn.FacetGrid(df, col='dataset')
g.map_dataframe(seaborn.regplot, x='x', y='y')

plt.show()