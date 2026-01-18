import pandas as pd

from ContextualAnomalyInject import GenerateData
from NS import NormalcyScore


def AverageTest(FilePath,
                ResultFilePath, 
                SaveFilePath_HGPR,
                MyColList, AllColsWithTruth, MyContextList, MyBehaveList, 
                NumCols, anomaly_value, sample_value, num_dataset):
    
    myResult_HGPR =  pd.DataFrame(columns = ["pr_auc_value","roc_auc_value"])
   
   
    for MyRandomState in range(42,42+num_dataset):

        FinalDataSet = GenerateData(FilePath, MyColList, MyContextList, MyBehaveList, NumCols, anomaly_value, MyRandomState)
        FinalDataSet.to_csv(ResultFilePath, sep=',')
        FinalDataSet = pd.read_csv(ResultFilePath, sep=",")
    
        FinalDataSet = FinalDataSet.dropna()
        
        MyDataSet = FinalDataSet[AllColsWithTruth]

    
        my_pr_auc, my_roc_score, P_at_n_value, TempDataSet = NormalcyScore(MyDataSet, AllColsWithTruth, MyContextList, MyBehaveList, sample_value)
        
        myResult_HGPR.loc[len(myResult_HGPR)] = [my_pr_auc, my_roc_score]

    myResult_HGPR.to_csv(SaveFilePath_HGPR, sep=',')
    

FilePath = 'datasets/abalone.csv'
ResultFilePath = 'abalone_injected.csv'
SaveFilePath_HGPR = 'abalone_NS_results.csv'


AllCols = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

AllColsWithTruth = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings',
                    'ground_truth']

ContextCols = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight']

BehaveCols = ['Rings']

NumCols = BehaveCols

anomaly_value = 100

sample_value = 100

num_dataset = 5


AverageTest(FilePath,
            ResultFilePath,
            SaveFilePath_HGPR,
            AllCols, AllColsWithTruth, ContextCols, BehaveCols, NumCols, 
            anomaly_value, sample_value, num_dataset)

