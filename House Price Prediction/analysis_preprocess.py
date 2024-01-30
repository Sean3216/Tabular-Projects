import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

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
    print("Number of columns:")
    print(len(df.columns))
    print("=====================================")
    print("Number of unique id:")
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
            #save high resolution image
            plt.savefig('correlationclustered.png', dpi=300)
        else:
            plt.figure(figsize=(20, 20))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f",vmin=-1, vmax=1)
            plt.savefig('correlation.png', dpi=300)

def checkallcatvaluecounts(df):
    for col in df.select_dtypes(include=['object']).columns:
        valco = df[col].value_counts(normalize=True).sort_values(ascending=False)
        print("=====================================")
        print(valco)
        print("=====================================")

def groupratings(x, low, high):
    if x < low:
        x = 'Low'
    elif x > high:
        x = 'High'
    else:
        x = 'Normal'
    return x


######################################################### PCA #########################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PrincipalComponentAnalysis():
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

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

        if mode == 'train':
            self.pca = PCA(n_components = n_components, svd_solver='full')
            self.pca.fit(scaleddata)
        if self.pca == None:
            print("Please train the model first")
            return None
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

###################################### Preprocessing Class ######################################
class Preprocessing():
    def __init__(self, df, test):
        self.tempdata = df.copy() #for fixing format

        self.data = df.copy() #traindata
        self.test = test.copy() #testdata
        self.PCA = PrincipalComponentAnalysis()

        #we need categorical data to be in the same format in both train and test data
        #so, assuming that we know the categories in the categorical features, set the categorical features before getting the dummies
        self.categories = {'MSSubClass': ['20','30','40','45','50','60','70','75','80','85','90','120','150','160','180','190'],
                           'MSZoning': ['A','C','FV','I','RH','RL','RP','RM','C (all)'],
                           'LotShape': ['Reg','IR1','IR2','IR3'],
                           'LandContour': ['Lvl','Bnk','HLS','Low'],
                           'LotConfig': ['Inside','Corner','CulDSac','FR2','FR3'],
                           'Neighborhood': ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel','Names','NoRidge','NPkVill','NridgHt','NWAmes','OldTown','SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker'],
                           'Condition1': ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe'],
                           'BldgType': ['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'],
                           'HouseStyle': ['1Story','1.5f','2Story','2.5f','SFoyer','SLvl'],
                           'RoofStyle': ['Flat','Gable','Gambrel','Hip','Mansard','Shed'],
                           'Exterior1st': ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing'],
                           'Exterior2nd': ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing'],
                           'MasVnrType': ['BrkCmn','BrkFace','CBlock','None','Stone'],
                           'ExterQual': ['Ex','Gd','TA','Fa','Po'],
                           'ExterCond': ['Ex','Gd','TA','Fa','Po'],
                           'Foundation': ['BrkTil','CBlock','PConc','Slab','Stone','Wood'],
                           'BsmtQual': ['Ex','Gd','TA','Fa','Po','NA'],
                           'BsmtExposure': ['Gd','Av','Mn','No','NA'],
                           'BsmtFinType1': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],
                           'BsmtFinType2': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA'],
                           'HeatingQC': ['Ex','Gd','TA','Fa','Po'],
                           'KitchenQual': ['Ex','Gd','TA','Fa','Po'],
                           'FireplaceQu': ['Ex','Gd','TA','Fa','Po','NA'],
                           'GarageType': ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA'],
                           'GarageFinish': ['Fin','RFn','Unf','NA'],
                           'SaleType': ['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth'],
                           'SaleCondition': ['Normal','Abnorml','AdjLand','Alloca','Family','Partial'],
                           'has2ndfloor': ['Yes','No'],
                           'hasfireplace': ['Yes','No'],
                           'multiplecarspace': ['Yes','No'],
                           'OverallQual': ['Low','Normal','High'],
                           'OverallCond': ['Low','Normal','High']}
                           

    def wrangletrain(self):
        self.data.drop(['Id'],axis = 1, inplace = True)

        ########### Numerical Var Preprocessing ###########
        #LotFrontage
        LFneighborhood = self.data.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        self.data['LotFrontage'] = self.data.apply(lambda row: LFneighborhood[row['Neighborhood']] if np.isnan(row['LotFrontage']) else row['LotFrontage'], axis=1)
        self.data['LotFrontage'] = self.data['LotFrontage']**2
        #MasVnrArea
        self.data['MasVnrArea'].fillna(0, inplace=True)

        #GarageYrBlt
        self.data['GarageYrBlt'].fillna(0, inplace=True)

        #Convert MSSubClass to categorical
        self.data['MSSubClass'] = self.data['MSSubClass'].astype('object')

        #### Feature Engineering ####
        ##This follows the reference provided by Ineeji
        
        #add more weight if the building is remodeled recently
        self.data['YrBltAndRemod']=self.data['YearBuilt']+self.data['YearRemodAdd']
        
        #captures the feeling of spaciness in the house
        self.data['TotalSF']=self.data['TotalBsmtSF'] + self.data['1stFlrSF'] + self.data['2ndFlrSF']
        self.data['Total_sqr_footage'] = (self.data['BsmtFinSF1'] + self.data['BsmtFinSF2'] +
                                        self.data['1stFlrSF'] + self.data['2ndFlrSF'])
        self.data['Total_porch_sf'] = (self.data['OpenPorchSF'] + self.data['3SsnPorch'] +
                                    self.data['EnclosedPorch'] + self.data['ScreenPorch'] +
                                    self.data['WoodDeckSF'])
        self.data['Total_Bathrooms'] = (self.data['FullBath'] + (0.5 * self.data['HalfBath']) +
                                    self.data['BsmtFullBath'] + (0.5 * self.data['BsmtHalfBath']))
        
        #Grouping ratings
        self.data['OverallQual'] = self.data['OverallQual'].apply(lambda x: groupratings(x, 5, 8)).astype('object')
        self.data['OverallCond'] = self.data['OverallCond'].apply(lambda x: groupratings(x, 5, 7)).astype('object')

        #Drop GarageArea. It's represented by GarageCars
        self.data.drop(columns = ['GarageArea'], inplace = True)
        
        ########### Categorical Var Preprocessing ###########
        columnstobedropped = ['PoolQC','MiscFeature','Alley','Fence','Street','Utilities',
                              'LandSlope','Condition2','RoofMatl','BsmtCond','Heating',
                              'CentralAir','Electrical','Functional','GarageQual',
                              'GarageCond','PavedDrive']
        self.data.drop(columnstobedropped,axis = 1, inplace = True)

        #filling missing values
        self.data['HouseStyle'] = self.data['HouseStyle'].apply(lambda x: '1.5f' if x == '1.5Fin' or x == '1.5Unf' else x)
        self.data['HouseStyle'] = self.data['HouseStyle'].apply(lambda x: '2.5f' if x == '2.5Fin' or x == '2.5Unf' else x)

        self.data.loc[(self.data['MasVnrType'].isna()) & (self.data['MasVnrArea'] > 0), 'MasVnrType'] = self.data['MasVnrType'].mode()[0]
        self.data['MasVnrType'].fillna('None',inplace = True)

        self.data[['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2',
                   'FireplaceQu','GarageType','GarageFinish']].fillna('NA',inplace = True)
        #dropping index 948
        print("Dropping index 948 because of this weird row")
        print("Value when BsmtExposure is NA but BsmtFinSF1 and BsmtUnfSF is not NA")
        print(self.data[self.data['BsmtExposure'].isna()][['BsmtFinSF1','BsmtUnfSF']].loc[948])
        self.data = self.data.drop(index = 948).reset_index(drop = True)

        #filling specific index on BsmtFinType2 with Unf
        print("Filling specific index (index 332) on BsmtFinType2 with 'Unf'")
        self.data.loc[332,'BsmtFinType2'] = 'Unf'

        #Feature Engineering
        self.data['has2ndfloor'] = self.data['2ndFlrSF'].apply(lambda i: "Yes" if i > 0 else "No").astype('object')
        self.data['hasfireplace'] = self.data['Fireplaces'].apply(lambda i: "Yes" if i > 0 else "No").astype('object')
        self.data['multiplecarspace'] = self.data['GarageCars'].apply(lambda i: "Yes" if i > 1 else "No").astype('object')
        
        ########### Finishing Preprocessing ###########
        target = self.data['SalePrice']

        numeric = self.PCA.do(self.data.select_dtypes(exclude = ['object']).drop(['SalePrice'], axis = 1),
                              n_components= 18, mode = 'train')

        categorical = self.data.select_dtypes(include = ['object'])
        categorical = self.objecttocategories(categorical)

        self.data = pd.concat([numeric,categorical,target],axis = 1)

        self.data = pd.get_dummies(self.data)
        return self.data
    
    def wrangletest(self):
        self.test.drop(['Id'],axis = 1, inplace = True)

        ########### Numerical Var Preprocessing ###########
        #LotFrontage
        LFneighborhood = self.test.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        self.test['LotFrontage'] = self.test.apply(lambda row: LFneighborhood[row['Neighborhood']] if np.isnan(row['LotFrontage']) else row['LotFrontage'], axis=1)
        self.test['LotFrontage'] = self.test['LotFrontage']**2

        #MasVnrArea
        self.test['MasVnrArea'].fillna(0, inplace=True)

        #GarageYrBlt
        self.test['GarageYrBlt'].fillna(0, inplace=True)

        #Convert MSSubClass to categorical
        self.test['MSSubClass'] = self.test['MSSubClass'].astype('object')

        #Before Feature Engineering, there might be missing values in other features that does not exist in the training set
        #accomodate them with KNNImputer
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        tempnumeric = self.test.select_dtypes(exclude = ['object'])
        tempnumericcol = tempnumeric.columns
        imputer.fit(self.test.select_dtypes(exclude = ['object']))
        numericimputed = pd.DataFrame(imputer.transform(tempnumeric),columns = tempnumericcol)
        self.test.drop(columns = tempnumericcol, inplace = True)
        self.test = pd.concat([self.test,numericimputed],axis = 1)

        #### Feature Engineering ####
        ##This follows the reference provided by Ineeji
        
        #add more weight if the building is remodeled recently
        self.test['YrBltAndRemod']=self.test['YearBuilt']+self.test['YearRemodAdd']
        
        #captures the feeling of spaciness in the house
        self.test['TotalSF']=self.test['TotalBsmtSF'] + self.test['1stFlrSF'] + self.test['2ndFlrSF']
        self.test['Total_sqr_footage'] = (self.test['BsmtFinSF1'] + self.test['BsmtFinSF2'] +
                                        self.test['1stFlrSF'] + self.test['2ndFlrSF'])
        self.test['Total_porch_sf'] = (self.test['OpenPorchSF'] + self.test['3SsnPorch'] +
                                    self.test['EnclosedPorch'] + self.test['ScreenPorch'] +
                                    self.test['WoodDeckSF'])
        self.test['Total_Bathrooms'] = (self.test['FullBath'] + (0.5 * self.test['HalfBath']) +
                                    self.test['BsmtFullBath'] + (0.5 * self.test['BsmtHalfBath']))
        
        #Grouping ratings
        self.test['OverallQual'] = self.test['OverallQual'].apply(lambda x: groupratings(x, 5, 8)).astype('object')
        self.test['OverallCond'] = self.test['OverallCond'].apply(lambda x: groupratings(x, 5, 7)).astype('object')

        #Drop GarageArea. It's represented by GarageCars
        self.test.drop(columns = ['GarageArea'], inplace = True)
        
        ########### Categorical Var Preprocessing ###########
        columnstobedropped = ['PoolQC','MiscFeature','Alley','Fence','Street','Utilities',
                              'LandSlope','Condition2','RoofMatl','BsmtCond','Heating',
                              'CentralAir','Electrical','Functional','GarageQual',
                              'GarageCond','PavedDrive']
        self.test.drop(columnstobedropped,axis = 1, inplace = True)

        self.test['HouseStyle'] = self.test['HouseStyle'].apply(lambda x: '1.5f' if x == '1.5Fin' or x == '1.5Unf' else x)
        self.test['HouseStyle'] = self.test['HouseStyle'].apply(lambda x: '2.5f' if x == '2.5Fin' or x == '2.5Unf' else x)
        #filling test data categorical features' missing values is also not known
        self.test.loc[(self.test['MasVnrType'].isna()) & (self.test['MasVnrArea'] > 0), 'MasVnrType'] = self.test['MasVnrType'].mode()[0]
        self.test['MasVnrType'].fillna('None',inplace = True)

        self.test[['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2',
                   'FireplaceQu','GarageType','GarageFinish']].fillna('NA',inplace = True)
        #fill other missing values with mode
        self.test.select_dtypes(include = ['object']).fillna(self.test.select_dtypes(include = ['object']).mode().iloc[0], inplace = True)

        #Feature Engineering
        self.test['has2ndfloor'] = self.test['2ndFlrSF'].apply(lambda i: "Yes" if i > 0 else "No").astype('object')
        self.test['hasfireplace'] = self.test['Fireplaces'].apply(lambda i: "Yes" if i > 0 else "No").astype('object')
        self.test['multiplecarspace'] = self.test['GarageCars'].apply(lambda i: "Yes" if i > 1 else "No").astype('object')

        ########### Finishing Preprocessing ###########
        #Fixing format
        self.fixingformat()

        numeric = self.PCA.do(self.test.select_dtypes(exclude = ['object']),
                              n_components= 18, mode = 'test')

        categorical = self.test.select_dtypes(include = ['object'])
        categorical = self.objecttocategories(categorical)
        
        self.test = pd.concat([numeric,categorical],axis = 1)

        self.test = pd.get_dummies(self.test)
        return self.test
    
    def objecttocategories(self, df):
        categoricalcolumns = self.categories.keys()
        df = df[categoricalcolumns]
        cat_as_list = [self.categories[col] for col in df.columns]
        OHC = OneHotEncoder(categories=cat_as_list, handle_unknown='ignore')
        OHC.fit(df)
        df = pd.DataFrame(OHC.transform(df).toarray(), columns=OHC.get_feature_names_out())
        return df
    
    def fixingformat(self):
        #we also need to make sure that the test data has the same datatype as the train data
        #so, we need to convert the datatype of the test data to the same as the train data
        exception = ['SalePrice','Id', 'TotalSF', 'Total_sqr_footage', 'Total_porch_sf', 'Total_Bathrooms', 'HasFireplace','Has2ndfloor','YrBltAndRemod',
                     'PoolQC','MiscFeature','Alley','Fence','Street','Utilities','LandSlope','Condition2','RoofMatl','BsmtCond','Heating','CentralAir',
                     'Electrical','Functional','GarageQual','GarageCond','PavedDrive','OverallQual','OverallCond', 'MSSubClass','GarageArea']
        for col in self.tempdata.columns:
            if col not in exception:            
                if self.tempdata[col].dtype != self.test[col].dtype:
                    print('fixing column {} in test data to fit format {}'.format(col,self.tempdata[col].dtype))
                    self.test[col] = self.test[col].astype(self.tempdata[col].dtype)
            else:
                pass

###################################### Feature Importance  ######################################
            
class FeatureImportance():
    def __init__(self,ridge, lasso, elasticnet, gbr, lightgbm, xgboost, df):
        self.ridge = ridge
        self.lasso = lasso
        self.elasticnet = elasticnet
        self.gbr = gbr
        self.lightgbm = lightgbm
        self.xgboost = xgboost
        self.df = df

    def calculate(self):
            #get feature names
            ridge_feature_names = self.df.columns
            lasso_feature_names = self.df.columns
            elasticnet_feature_names = self.df.columns
            gbr_feature_names = self.df.columns
            lightgbm_feature_names = self.df.columns
            xgboost_feature_names = self.df.columns
            #get the feature importance of each model as dictionary and set keys to be the feature names
            ridge_feature_importance = dict(zip(ridge_feature_names, self.ridge.coef_))
            lasso_feature_importance = dict(zip(lasso_feature_names, self.lasso.coef_))
            elasticnet_feature_importance = dict(zip(elasticnet_feature_names, self.elasticnet.coef_))
            gbr_feature_importance = dict(zip(gbr_feature_names, self.gbr.feature_importances_))
            lightgbm_feature_importance = dict(zip(lightgbm_feature_names, self.lightgbm.feature_importances_))
            xgboost_feature_importance = dict(zip(xgboost_feature_names, self.xgboost.feature_importances_))

            #Distance-based models often times have positive and negative feature importance while tree-based models only have positive feature importance
            #so, we need to make sure that the feature importance of distance-based models are all positive
            ridge_feature_importance = {k:abs(v) for k,v in ridge_feature_importance.items()}
            lasso_feature_importance = {k:abs(v) for k,v in lasso_feature_importance.items()}
            elasticnet_feature_importance = {k:abs(v) for k,v in elasticnet_feature_importance.items()}

            #make a dataframe with feature names and feature importance of each model as columns
            df_importance = pd.DataFrame()
            df_importance['Feature'] = ridge_feature_importance.keys()
            for feature in ridge_feature_importance.keys():
                df_importance.loc[df_importance['Feature'] == feature, 'Ridge'] = ridge_feature_importance[feature]
                df_importance.loc[df_importance['Feature'] == feature, 'Lasso'] = lasso_feature_importance[feature]
                df_importance.loc[df_importance['Feature'] == feature, 'ElasticNet'] = elasticnet_feature_importance[feature]
                df_importance.loc[df_importance['Feature'] == feature, 'GradientBoosting'] = gbr_feature_importance[feature]
                df_importance.loc[df_importance['Feature'] == feature, 'LightGBM'] = lightgbm_feature_importance[feature]
                df_importance.loc[df_importance['Feature'] == feature, 'XGBoost'] = xgboost_feature_importance[feature]
            return df_importance

    def getfeatureimportance(self,weightridge = 1, weightlasso = 1, weightelasticnet = 1, weightgbr = 1, weightlightgbm = 1, weightxgboost = 1):
        importance = self.calculate()
        importance['Importance'] = (
            importance['Ridge']*weightridge + importance['Lasso']*weightlasso + importance['ElasticNet']*weightelasticnet + 
            importance['GradientBoosting']*weightgbr + importance['LightGBM']*weightlightgbm + importance['XGBoost']*weightxgboost
            )
        importance.drop(columns = ['Ridge','Lasso','ElasticNet','GradientBoosting','LightGBM','XGBoost'], inplace = True)
        importance.sort_values(by = 'Importance', ascending = False, inplace = True)
        return importance
    
    def plotfeatureimportance(self,importancedf):
        plt.figure(figsize=(20, 20))
        sns.barplot(x="Importance", y="Feature", data=importancedf)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show(block=False)

###################################### Realtor's Price Predictor ######################################
def realtorpredict(df):
    '''
    Group the sale price by Neighborhood, MSZoning, Condition1, Condition2, 
    BldgType, HouseStyle, Utilities, BsmtCond, GarageCond, SaleType, and SaleCondition 
    then get the mean.
    '''
    #Select respective columns
    predictorcol = ['Neighborhood','MSZoning','Condition1','Condition2','BldgType','HouseStyle',
                    'Utilities', 'BsmtCond','GarageCond','SaleType','SaleCondition']
    df = df[predictorcol+['SalePrice']]
    #Fill missing values with NA
    df['BsmtCond'] = df['BsmtCond'].fillna('NA')
    df['GarageCond'] = df['GarageCond'].fillna('NA')

    #drop index 948
    df = df.drop(index = 948).reset_index(drop = True)

    #Group by the columns and get the mean
    priceintuition = df.groupby(predictorcol).mean()

    #create a new column in the df named 'PriceIntuition' and fill with mean from groupby
    df_predicted = pd.merge(df, priceintuition, on=predictorcol, how='left', suffixes=('', '_priceintuition'))

    # Fill missing values in case some combinations didn't have mean prices (optional)
    df_predicted['SalePrice_priceintuition'] = df_predicted['SalePrice_priceintuition'].fillna(priceintuition['SalePrice'].mean())

    # Predict SalePrice using the mean prices
    df_predicted['PredictedSalePrice'] = df_predicted['SalePrice_priceintuition']

    # Drop the mean price column if not needed
    df_predicted.drop(columns='SalePrice_priceintuition', inplace=True)

    return df_predicted

    