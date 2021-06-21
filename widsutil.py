from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
import os, gc, sys, time, random, math
from scipy import stats, special
from functools import partial
import pandas as pd
import typing as tp
import numpy as np


## This file has DataPreprocess class with all preprocessing utility functions which can be used
## for train, test data preprocessing for WIDS ICU dataset


class DataPreprocess:
    
    def __init__(self, data):
        self.df= data
     
    
    
    def dataprep(self, train):
        #if(train['pao2fio2ratio_apache']):
        train = train.rename(columns={'pao2_apache':'pao2fio2ratio_apache','ph_apache':'arterial_ph_apache'})
        train.loc[train.age == 0, 'age'] = np.nan
        train = train.drop(['readmission_status','encounter_id','hospital_id'], axis=1)
        train = train.replace([np.inf, -np.inf], np.nan)
        #min max value collector
        min_max_feats=[f[:-4] for f in train.columns if f[-4:]=='_min']
        for col in min_max_feats:
            train.loc[train[f'{col}_min'] > train[f'{col}_max'], [f'{col}_min', f'{col}_max']] = train.loc[train[f'{col}_min'] > train[f'{col}_max'], [f'{col}_max', f'{col}_min']].values
        #print the missing count 
        print(f'Percent of Nans in Train Data : {round(train.isna().sum().sum()/len(train), 2)}')
        return train
    
    #encoding
    def lblencoder(self, train):
        lbls = {}
        for col in train.select_dtypes(exclude = np.number).columns.tolist():
            le = LabelEncoder().fit(pd.concat([train[col].astype(str)]))   
            train[col] = le.transform(train[col].astype(str))
            lbls[col] = le
        print('Categorical columns:', list(lbls.keys()))
        return train
    
    # transform function 
    def datatransform(self, train):
       #transformation
        train['comorbidity_score'] = train['aids'].values * 23 + train['cirrhosis'] * 4  + train['hepatic_failure'] * 16 + train['immunosuppression'] * 10 + train['leukemia'] * 10 + train['lymphoma'] * 13 + train['solid_tumor_with_metastasis'] * 11
        train['comorbidity_score'] = train['comorbidity_score'].fillna(0)
        train['gcs_sum'] = train['gcs_eyes_apache']+train['gcs_motor_apache']+train['gcs_verbal_apache']
        train['gcs_sum'] = train['gcs_sum'].fillna(0)
        train['apache_2_diagnosis_type'] = train.apache_2_diagnosis.round(-1).fillna(-100).astype('int32')
        train['apache_3j_diagnosis_type'] = train.apache_3j_diagnosis.round(-2).fillna(-100).astype('int32')
        train['bmi_type'] = train.bmi.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
        train['height_type'] = train.height.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
        train['weight_type'] = train.weight.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
        train['age_type'] = train.age.fillna(0).apply(lambda x: 10 * (round(int(x)/10)))
        train['gcs_sum_type'] = train.gcs_sum.fillna(0).apply(lambda x: 2.5 * (round(int(x)/2.5))).divide(2.5)
        train['apache_3j_diagnosis_x'] = train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
        train['apache_2_diagnosis_x'] = train['apache_2_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
        #train['apache_3j_diagnosis_split1'] = np.where(train['apache_3j_diagnosis'].isna() , np.nan , train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[1]  )
        #train['apache_2_diagnosis_split1'] = np.where(train['apache_2_diagnosis'].isna() , np.nan , train['apache_2_diagnosis'].apply(lambda x : x % 10)  )
    
        train['apache_3j_diagnosis_split1'] = np.where(train['apache_3j_diagnosis'].isna() , 0 , train['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[1]  )
        train['apache_2_diagnosis_split1'] = np.where(train['apache_2_diagnosis'].isna() , 0 , train['apache_2_diagnosis'].apply(lambda x : x % 10)  )
    
        IDENTIFYING_COLS = ['age_type', 'height_type',  'ethnicity', 'gender', 'bmi_type'] 
        train['profile'] = train[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis = 1)
        print(f'Number of unique Profiles : {train["profile"].nunique()}')
        #BMI transforation
        train["diff_bmi"] = train['bmi'].copy() 
        train['bmi'] = train['weight']/((train['height']/100)**2)
        train["diff_bmi"] = train["diff_bmi"]-train['bmi']
        train['pre_icu_los_days'] = train['pre_icu_los_days'].apply(lambda x:special.expit(x) )
        train['abmi'] = train['age']/train['bmi']
        train['agi'] = train['weight']/train['age']
        # daily and Hourly labstests columns transformation
        d_cols = [c for c in train.columns if(c.startswith("d1"))]
        h_cols = [c for c in train.columns if(c.startswith("h1"))]
        train["dailyLabs_row_nan_count"] = train[d_cols].isna().sum(axis=1)
        train["hourlyLabs_row_nan_count"] = train[h_cols].isna().sum(axis=1)
        train["diff_labTestsRun_daily_hourly"] = train["dailyLabs_row_nan_count"] - train["hourlyLabs_row_nan_count"]
    
        return train
        
    def labtesttransform(self, train):
        lab_col = [c for c in train.columns if((c.startswith("h1")) | (c.startswith("d1")))]
        lab_col_names = list(set(list(map(lambda i: i[ 3 : -4], lab_col))))
        print("len lab_col",len(lab_col))
        print("len lab_col_names",len(lab_col_names))
        print("lab_col_names\n",lab_col_names)
        first_h = []
        print()
        for v in lab_col_names:
            first_h.append(v+"_started_after_firstHour")
            #colsx = [x for x in test_df.columns if v in x]
            #colsx = train_df.columns
            #print(train.loc[:, colsx].isna().sum(axis=1))
                
            #train[v+"_nans"] = train.loc[:, colsx].isna().sum(axis=1)
            train[v+"_d1_value_range"] = train[f"d1_{v}_max"].subtract(train[f"d1_{v}_min"])
            train[v+"_h1_value_range"] = train[f"h1_{v}_max"].subtract(train[f"h1_{v}_min"])
            train[v+"_d1_h1_max_eq"] = (train[f"d1_{v}_max"]== train[f"h1_{v}_max"]).astype(np.int8)
            train[v+"_d1_h1_min_eq"] = (train[f"d1_{v}_min"]== train[f"h1_{v}_min"]).astype(np.int8)
            train[v+"_d1_zero_range"] = (train[v+"_d1_value_range"] == 0).astype(np.int8)
            train[v+"_h1_zero_range"] =(train[v+"_h1_value_range"] == 0).astype(np.int8)
            train[v+"_tot_change_value_range_normed"] = abs((train[v+"_d1_value_range"].div(train[v+"_h1_value_range"])))#.div(df[f"d1_{v}_max"]))
            train[v+"_started_after_firstHour"] = ((train[f"h1_{v}_max"].isna()) & (train[f"h1_{v}_min"].isna())) & (~train[f"d1_{v}_max"].isna())
            train[v+"_day_more_extreme"] = ((train[f"d1_{v}_max"]>train[f"h1_{v}_max"]) | (train[f"d1_{v}_min"]<train[f"h1_{v}_min"]))
            train[v+"_day_more_extreme"].fillna(False)               
        train["total_Tests_started_After_firstHour"] = train[first_h].sum(axis=1)
        gc.collect()
        train["total_Tests_started_After_firstHour"].describe()
            
        return train

    def dataparametertfm(self, train):
        train['diasbp_indicator'] = (
        (train['d1_diasbp_invasive_max'] == train['d1_diasbp_max']) & (train['d1_diasbp_noninvasive_max']==train['d1_diasbp_invasive_max'])|
        (train['d1_diasbp_invasive_min'] == train['d1_diasbp_min']) & (train['d1_diasbp_noninvasive_min']==train['d1_diasbp_invasive_min'])|
        (train['h1_diasbp_invasive_max'] == train['h1_diasbp_max']) & (train['h1_diasbp_noninvasive_max']==train['h1_diasbp_invasive_max'])|
        (train['h1_diasbp_invasive_min'] == train['h1_diasbp_min']) & (train['h1_diasbp_noninvasive_min']==train['h1_diasbp_invasive_min'])
        ).astype(np.int8)


        train['mbp_indicator'] = (
        (train['d1_mbp_invasive_max'] == train['d1_mbp_max']) & (train['d1_mbp_noninvasive_max']==train['d1_mbp_invasive_max'])|
        (train['d1_mbp_invasive_min'] == train['d1_mbp_min']) & (train['d1_mbp_noninvasive_min']==train['d1_mbp_invasive_min'])|
        (train['h1_mbp_invasive_max'] == train['h1_mbp_max']) & (train['h1_mbp_noninvasive_max']==train['h1_mbp_invasive_max'])|
        (train['h1_mbp_invasive_min'] == train['h1_mbp_min']) & (train['h1_mbp_noninvasive_min']==train['h1_mbp_invasive_min'])
        ).astype(np.int8)

        train['sysbp_indicator'] = (
        (train['d1_sysbp_invasive_max'] == train['d1_sysbp_max']) & (train['d1_sysbp_noninvasive_max']==train['d1_sysbp_invasive_max'])|
        (train['d1_sysbp_invasive_min'] == train['d1_sysbp_min']) & (train['d1_sysbp_noninvasive_min']==train['d1_sysbp_invasive_min'])|
         (train['h1_sysbp_invasive_max'] == train['h1_sysbp_max']) & (train['h1_sysbp_noninvasive_max']==train['h1_sysbp_invasive_max'])|
        (train['h1_sysbp_invasive_min'] == train['h1_sysbp_min']) & (train['h1_sysbp_noninvasive_min']==train['h1_sysbp_invasive_min'])
        ).astype(np.int8)
        
        
        
        train['d1_mbp_invnoninv_max_diff'] = train['d1_mbp_invasive_max'] - train['d1_mbp_noninvasive_max']
        train['h1_mbp_invnoninv_max_diff'] = train['h1_mbp_invasive_max'] - train['h1_mbp_noninvasive_max']
        train['d1_mbp_invnoninv_min_diff'] = train['d1_mbp_invasive_min'] - train['d1_mbp_noninvasive_min']
        train['h1_mbp_invnoninv_min_diff'] = train['h1_mbp_invasive_min'] - train['h1_mbp_noninvasive_min']
        train['d1_diasbp_invnoninv_max_diff'] = train['d1_diasbp_invasive_max'] - train['d1_diasbp_noninvasive_max']
        train['h1_diasbp_invnoninv_max_diff'] = train['h1_diasbp_invasive_max'] - train['h1_diasbp_noninvasive_max']
        train['d1_diasbp_invnoninv_min_diff'] = train['d1_diasbp_invasive_min'] - train['d1_diasbp_noninvasive_min']
        train['h1_diasbp_invnoninv_min_diff'] = train['h1_diasbp_invasive_min'] - train['h1_diasbp_noninvasive_min']
        train['d1_sysbp_invnoninv_max_diff'] = train['d1_sysbp_invasive_max'] - train['d1_sysbp_noninvasive_max']
        train['h1_sysbp_invnoninv_max_diff'] = train['h1_sysbp_invasive_max'] - train['h1_sysbp_noninvasive_max']
        train['d1_sysbp_invnoninv_min_diff'] = train['d1_sysbp_invasive_min'] - train['d1_sysbp_noninvasive_min']
        train['h1_sysbp_invnoninv_min_diff'] = train['h1_sysbp_invasive_min'] - train['h1_sysbp_noninvasive_min']
        
        for v in ['albumin','bilirubin','bun','glucose','hematocrit','pao2fio2ratio','arterial_ph','resprate','sodium','temp','wbc','creatinine']:
            train[f'{v}_indicator'] = (((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_max']) & (train[f'd1_{v}_max']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'd1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_max']) & (train[f'h1_{v}_max']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'd1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'h1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'd1_{v}_min']) & (train[f'd1_{v}_min']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'h1_{v}_max'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'd1_{v}_min'])) |
                     ((train[f'{v}_apache']==train[f'h1_{v}_min']) & (train[f'h1_{v}_min']==train[f'd1_{v}_max']))
                    ).astype(np.int8)
        return train

           
               
    def dataoutliertfm(self, train):
        more_extreme_cols = [c for c in train.columns if(c.endswith("_day_more_extreme"))]
        train["total_day_more_extreme"] = train[more_extreme_cols].sum(axis=1)
        train["d1_resprate_div_mbp_min"] = train["d1_resprate_min"].div(train["d1_mbp_min"])
        train["d1_resprate_div_sysbp_min"] = train["d1_resprate_min"].div(train["d1_sysbp_min"])
        train["d1_lactate_min_div_diasbp_min"] = train["d1_lactate_min"].div(train["d1_diasbp_min"])
        train["d1_heartrate_min_div_d1_sysbp_min"] = train["d1_heartrate_min"].div(train["d1_sysbp_min"])
        train["d1_hco3_div"]= train["d1_hco3_max"].div(train["d1_hco3_min"])
        train["d1_resprate_times_resprate"] = train["d1_resprate_min"].multiply(train["d1_resprate_max"])
        train["left_average_spo2"] = (2*train["d1_spo2_max"] + train["d1_spo2_min"])/3
        train["total_chronic"] = train[["aids","cirrhosis", 'hepatic_failure']].sum(axis=1)
        train["total_cancer_immuno"] = train[[ 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].sum(axis=1)
        train["has_complicator"] = train[["aids","cirrhosis", 'hepatic_failure',
                                'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].max(axis=1)
        
        train[["has_complicator","total_chronic","total_cancer_immuno","has_complicator"]].describe()
        #missing values
        train['apache_3j'] = np.where(train['apache_3j_diagnosis_type']<0 , np.nan ,
                                np.where(train['apache_3j_diagnosis_type'] < 200, 'Cardiovascular' ,
                                np.where(train['apache_3j_diagnosis_type'] < 400, 'Respiratory' ,
                                np.where(train['apache_3j_diagnosis_type'] < 500, 'Neurological' ,
                                np.where(train['apache_3j_diagnosis_type'] < 600, 'Sepsis' ,
                                np.where(train['apache_3j_diagnosis_type'] < 800, 'Trauma' ,
                                np.where(train['apache_3j_diagnosis_type'] < 900, 'Haematological' ,       
                                np.where(train['apache_3j_diagnosis_type'] < 1000, 'Renal/Genitourinary' , 
                                np.where(train['apache_3j_diagnosis_type'] < 1200, 'Musculoskeletal/Skin disease' , 'Operative Sub-Diagnosis Codes' ))))))))
                                        )
        cols = ['apache_3j_diagnosis_x', 'apache_2_diagnosis_x', 'apache_3j_diagnosis_split1', 'apache_3j']
        for i in cols:
            train[i] = pd.to_numeric(train[i],errors='coerce')
        gc.collect()
        return train

  
    def preprocess(self):
        self.df = self.dataprep(self.df)
        self.df = self.lblencoder(self.df)
        self.df =self.datatransform(self.df)
        self.df =self.labtesttransform(self.df)
        self.df =self.dataparametertfm(self.df)
        self.df = self.dataoutliertfm(self.df)
        return self.df
  
