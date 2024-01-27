import pandas as pd 
import numpy as np 

from scipy.stats import chi2_contingency
from scipy.stats import chi2

import matplotlib.pyplot as plt
import seaborn as sns

def describe(df):
    print(df.info())
    print("=====================================")
    print("Number of rows:")
    print(len(df))
    print("=====================================")
    print("Number if unique id:")
    print(df['Id'].nunique())
    print("=====================================")
    print("Number of unique houses:")
    print(len(df.drop(['Id','SalePrice','SaleCondition','SaleType'],axis = 1).drop_duplicates()))
    print("=====================================")
    print(df.sample(10))
    #shuffle all row df while setting seed to 42
    


def checkmissing(df, threshold = 0.3, save = False):
    numofmv = df.isnull().sum()/len(df)
    print('Columns that has missing values:\n{}'.format(numofmv[numofmv>0]))
    if threshold > 0:
        print('Columns that has large missing values:\n{}'.format(numofmv[numofmv>threshold].sort_values(ascending=False)))
    if save:
        print("Return columns that has large missing values (above {}%)".format(threshold*100))
        return numofmv[numofmv>threshold]

def checkcorrelation(df, target=None, threshold = None, cluster = False):
    correlation_matrix = df.corr()
    if target != None:
        correlation_matrix = correlation_matrix[[target]].sort_values(by=target, ascending=False)
        if threshold != None:    
            correlation_matrix = correlation_matrix[abs(correlation_matrix) > threshold]
        print(correlation_matrix)
    else:
        if cluster:
            sns.clustermap(correlation_matrix, annot=True, fmt=".2f", figsize=(20, 20),vmin=-1, vmax=1)
        else:
            plt.figure(figsize=(20, 20))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f",vmin=-1, vmax=1)

def checkallcatvaluecounts(df):
    for col in df.select_dtypes(include=['object']).columns:
        valco = df[col].value_counts(normalize=True).sort_values(ascending=False)
        print("=====================================")
        print(valco)
        print("=====================================")


######################################################### PCA #########################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class PrincipalComponentAnalysis():
    def __init__(self):
        self.scaler = StandardScaler()

    def explain(self, df):
        scaleddata = self.scaler.fit_transform(df)
        pca = PCA(svd_solver='full')
        pca.fit(scaleddata)
        
        print(pca.explained_variance_ratio_.cumsum())
        scree = pca.explained_variance_ratio_*100
        plt.bar(np.arange(len(scree))+1, scree)
        plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Percentage Explained Variance")
        plt.title("Scree Plot")
        plt.show(block=False)

        print("=====================================")
        print("Number of Principal Components that explains 85% of variance:")
        num = np.where(pca.explained_variance_ratio_.cumsum() > 0.85)[0][0]+1
        print(num)
        print("=====================================")
        pcanew = PCA(n_components = num,svd_solver = 'full')
        pcanew.fit(scaleddata)
        loadings = pd.DataFrame(pcanew.components_.T, columns=["PC_{}".format(i+1) for i in range(num)], index=df.columns)
        loadings[(loadings>0.3)|(loadings<-0.3)]
        print("Loadings:")
        print(loadings)


    def do(self, df, n_components = None, mode = 'train'):
        if mode == 'train':
            self.scaler.fit(df)
        scaleddata = self.scaler.transform(df)

        if n_components == None:
            n_components = len(df.columns)
        self.pca = PCA(n_components = n_components, svd_solver='full')

        if mode == 'train':
            self.pca.fit(scaleddata)
        rawoutput = self.pca.transform(scaleddata)
        output = pd.DataFrame(rawoutput, columns=["PC_{}".format(i+1) for i in range(n_components)])
        return output

######################################################### MCA #########################################################
from prince import MCA

class MultipleCorrespondenceAnalysis():
    def explain(self, df):
        mca = MCA(n_components = len(df.columns),random_state = 42)
        mca.fit(df)
        
        print(mca.eigenvalues_summary)
        percentvar = mca.eigenvalues_summary['% of variance'].tolist()
        cumvar = mca.eigenvalues_summary['% of variance (cumulative)'].tolist()
        #remove "%" from the list
        percentvar = [x[:-1] for x in percentvar]
        cumvar = [x[:-1] for x in cumvar]
        #convert to float
        percentvar = [float(x)/100 for x in percentvar]
        cumvar = sorted([float(x)/100 for x in cumvar])

        plt.bar(np.arange(len(percentvar))+1, percentvar)
        plt.plot(np.arange(len(percentvar))+1, cumvar,c="red",marker='o')
        plt.xlabel("Number of Components")
        plt.ylabel("Percentage Explained Variance")
        plt.title("Scree Plot")
        plt.show(block=False)

        print("=====================================")
        print("Number of Components that explains 85% of variance:")
        num = len([x for x in cumvar if x <= 0.85])
        print("=====================================")
        mcanew = MCA(n_components = num,random_state = 42)
        mcanew.fit(df)
        print("Coordinates (MCA equivalent of loadings):")
        print(mcanew.row_coordinates(df))

    def do(self, df, n_components = None, mode = 'train'):
        if n_components == None:
            n_components = len(df.columns)
        self.mca = MCA(n_components = n_components, random_state = 42)

        if mode == 'train':
            self.mca.fit(df)
        rawoutput = self.mca.transform(df)
        output = pd.DataFrame(rawoutput, columns=["Coord_{}".format(i+1) for i in range(n_components)])
        return output


###################################### Outlier Elimination by Iterative IQR ######################################
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound))

def outliersearch(df, threshold = 3):
    rep_index = [] #this list will contain the indexes of rows considered as outlier for each columns
    for i in range(df.shape[1]): #for each columns in the dataset
        #append the indexes of rows considered as outlier of each columns to rep_index 
        rep_index.append(outliers_iqr(df.iloc[:,i]))
        
    a=0 
    outlier_index = []
    for i in range(len(df)): #for rows in the data
        a=0
        for j in rep_index: 
            #a single datapoint in rep_index contains the row indexes of a certain columns
            #if the current row index is within this single datapoint in rep_index, plus 1
            if i in j[0]: 
                a+=1
        if a>threshold: 
            #if the number of columns that the current row is considered as outlier 
            #is more than threshold, append the row index to outlier_index
            outlier_index.append(i) 
        
    return outlier_index