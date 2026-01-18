import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy
import random

def GenerateData(FilePath, AllCols, ContextCols, BehaveCols, NumCols, OutlierNum = 50, RandomState = 42):
    
    numpy.random.seed(RandomState)
    random.seed(RandomState)
    
    RawDataSet = pd.read_csv(FilePath, sep=",")
    
    RawDataSet = RawDataSet.dropna()
    
    MyDataSet = RawDataSet[AllCols]
                
    MyScaler = MinMaxScaler()
    
    Behavioural_list = BehaveCols

    if len(NumCols) > 0:
        MyDataSet[NumCols] = MyScaler.fit_transform(MyDataSet[NumCols])
            
    random.seed(RandomState)
    
    random_indices = random.sample(range(1, MyDataSet.shape[0]), OutlierNum)
    
    print(random_indices)
    
    for num_index in random_indices:
        print(num_index)  
    
        alpha_value = 0.5
        
        for col_index in Behavioural_list:
            luck_number = random.choice([-1,1])* random.uniform(0.1, alpha_value)
            MyDataSet[col_index].iloc[num_index] = MyDataSet[col_index].iloc[num_index] + luck_number

        print(MyDataSet[Behavioural_list].iloc[num_index])
       
    MyDataSet["ground_truth"] = 0
    
    for num_index in random_indices:
        MyDataSet["ground_truth"].iloc[num_index] = 1
    
    
    return MyDataSet
