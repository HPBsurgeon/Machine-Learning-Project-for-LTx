import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from statistics import stdev

Df = pd.read_excel('/Users/yanagawarintaro/Desktop/research/re_DRI_study/DRI.xlsx')
df = Df[(Df['TX_YEAR'] >= 2010) & (Df['TX_YEAR'] <= 2020)]
df = df[df['AGE'] >= 18]
# df = df[~(df['PREV_TX'] == 'Y')]
df = df[~(df['WLHR'] == 'Y')]
df = df[~(df['WLHL'] == 'Y')]
df = df[~(df['WLIN'] == 'Y')]
df = df[~(df['WLKI'] == 'Y')]
df = df[~(df['WLKP'] == 'Y')]
df = df[~(df['WLLU'] == 'Y')]
df = df[~(df['WLPA'] == 'Y')]
df = df[~(df['WLPI'] == 'Y')]
df = df[~(df['WLVC'] == 'Y')]
df = df[~(df['DON_TY'] == 'L')]

df["AGE_DON"] = df["AGE_DON"].fillna(df["AGE_DON"].median())
df["BMI_DON_CALC"] = df["BMI_DON_CALC"].fillna(df["BMI_DON_CALC"].median())
df["HGT_CM_DON_CALC"] = df["HGT_CM_DON_CALC"].fillna(df["HGT_CM_DON_CALC"].median())
df["DISTANCE"] = df["DISTANCE"].fillna(df["DISTANCE"].median())
df["COLD_ISCH"] = df["COLD_ISCH"].fillna(df["COLD_ISCH"].median())
df["TBILI_DON"] = df["TBILI_DON"].fillna(df["TBILI_DON"].median())
df["SGOT_DON"] = df["SGOT_DON"].fillna(df["SGOT_DON"].median())
df["SGPT_DON"] = df["SGPT_DON"].fillna(df["SGPT_DON"].median())
df["CREAT_DON"] = df["CREAT_DON"].fillna(df["CREAT_DON"].median())

age = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('AGE')]
    if 0 <= a < 18:
        x = np.NAN
    elif 18 <= a < 39:
        x = 0
    elif 40 <= a < 49:
        x = 1
    elif 50 <= a < 59:
        x = 2
    elif 60 <= a < 69:
        x = 3
    elif 70 <= a:
        x = 4
    else:
        x = np.NAN #NAN
    age.append(x)

df['AGE'] = age


GENDER = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('GENDER')]
    if a == 'M':
        x = 0
    elif a == 'F':
        x = 1
    else:
        x = np.NAN
    GENDER.append(x)

df['GENDER'] = GENDER

MED_COND_TRR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('MED_COND_TRR')]
    if a == 3:
        x = 0
    elif a == 2:
        x = 1
    elif a == 1:
        x = 2
    else:
        x = np.NAN
    MED_COND_TRR.append(x)

df['MED_COND_TRR'] = MED_COND_TRR

PREV_TX = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PREV_TX')]
    if a == 'Y':
        x = 1
    else:
        x = 0
    PREV_TX.append(x)

df['PREV_TX'] = PREV_TX

BMI_REC = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('BMI_CALC')]
    if 0 <= a < 35:
        x = 0
    elif 35 <= a < 40:
        x = 1
    elif 40 <= a:
        x = 2
    else:
        x = np.NAN #NAN
    BMI_REC.append(x)

df['BMI_REC'] = BMI_REC

FINAL_SERUM_SODIUM = []
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('FINAL_SERUM_SODIUM')]
    if a >= 137:
        x = 137
    elif a <= 125:
        x = 125
    else:
        x = a 
    FINAL_SERUM_SODIUM.append(x)

df['FINAL_SERUM_SODIUM'] = FINAL_SERUM_SODIUM

df['MELD_NA'] = df['FINAL_MELD_PELD_LAB_SCORE'] + 1.32*(137-df['FINAL_SERUM_SODIUM']) - 0.033 * df['FINAL_MELD_PELD_LAB_SCORE'] * (137 - df['FINAL_SERUM_SODIUM'])

ASCITES_TX = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ASCITES_TX')]
    if a == 1:
        x = 0
    elif a == 2:
        x = 1
    elif a == 3:
        x = 2
    else:
        x = np.NAN
    ASCITES_TX.append(x)

df['ASCITES_TX'] = ASCITES_TX

ALBUMIN_TX = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ALBUMIN_TX')]
    if 0 <= a < 2.0:
        x = 2
    elif 2.0 <= a < 2.5:
        x = 1
    elif 2.5 <= a:
        x = 0
    else:
        x = np.NAN #NAN
    ALBUMIN_TX.append(x)

df['ALBUMIN_TX'] = ALBUMIN_TX

df['TBILI_TX']

CREAT_TX = []#連続で
for i in range(len(df)):  
    a = df.iat[i, df.columns.get_loc('CREAT_TX')]
    if a <= 1.5:
        x = 0
    elif a > 1.5:
        x = 1
    else:
        x = np.NAN #NAN
    CREAT_TX.append(x)

df['CREAT_TX'] = CREAT_TX

DIAB = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIAB')]
    if a == 1:
        x = 0
    elif a == 2:
        x = 1
    elif a == 3:
        x = 1
    elif a == 4:
        x = 1
    else:
        x = np.NAN
    DIAB.append(x)

df['DIAB'] = DIAB

df['HGT_REC']= (df['HGT_CM_CALC']-170)/10

AHN = []
Cholestatic_disease = []
Malignant = []
HCV = []
AIH_PBC_PSC = []
# REC_other_diag = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIAG')]
    if a == 4100:
        x = 1
    elif a == 4101:
        x = 1
    elif a == 4102:
        x = 1
    elif a == 4103:
        x = 1
    elif a == 4104:
        x = 1  
    elif a == 4105:
        x = 1
    elif a == 4106:
        x = 1
    elif a == 4107:
        x = 1
    elif a == 4108:
        x = 1
    elif a == 4109:
        x = 1    
    elif a == 4110:
        x = 1  
    else:
        x = 0
    AHN.append(x)

df['AHN'] = AHN
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIAG')]
    if a == 4220:
        immu = 1
    elif a == 4240:
        immu = 1
    elif a == 4241:
        immu = 1
    elif a == 4242:
        immu = 1
    elif a == 4245:
        immu = 1
    elif a == 4212:
        immu = 1 
    else:
        immu = 0
    AIH_PBC_PSC.append(immu)

df['AIH_PBC_PSC'] = AIH_PBC_PSC

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIAG')]
    if a == 4250:
        x = 1
    elif a == 4255:
        x = 1
    elif a == 4260:
        x = 1
    elif a == 4264:
        x = 1
    else:
        x = 0
    Cholestatic_disease.append(x)

df['Cholestatic_disease'] = Cholestatic_disease

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIAG')]
    if a == 4204:
        x = 1
    # elif a == 4206:
    #     x = 1
    elif a == 4216:
        x = 1
    elif a == 4593:
        x = 1
    elif a == 4104:
        x = 1
    else:
        x = 0
    HCV.append(x)

df['HCV'] = HCV

Life_support = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('LIFE_SUP_TRR')]
    b = df.iat[i, df.columns.get_loc('OTH_LIFE_SUP_TRR')]
    if a == 'Y':
        x = 1
    elif b == 1:
        x = 1
    else:
        x = 0
    Life_support.append(x)

df['Life_support'] = Life_support

PREV_AB_SURG_TCR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PREV_AB_SURG_TCR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    PREV_AB_SURG_TCR.append(x)

df['PREV_AB_SURG_TCR'] = PREV_AB_SURG_TCR

df['ON_VENT_TRR']

PORTAL_VEIN_TRR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PORTAL_VEIN_TRR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    PORTAL_VEIN_TRR.append(x)

df['PORTAL_VEIN_TRR'] = PORTAL_VEIN_TRR

MALIG = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('MALIG')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    MALIG.append(x)

df['MALIG'] = MALIG


TATTOOS = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TATTOOS')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    TATTOOS.append(x)

df['TATTOOS'] = TATTOOS

PROTEIN_URINE = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PROTEIN_URINE')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    PROTEIN_URINE.append(x)

df['PROTEIN_URINE'] = PROTEIN_URINE

CARDARREST_NEURO = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('CARDARREST_NEURO')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    CARDARREST_NEURO.append(x)

df['CARDARREST_NEURO'] = CARDARREST_NEURO

TIPSS = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TIPSS_TRR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0.5 ######
    else:
        x = np.NAN
    TIPSS.append(x)

df['TIPSS_TRR'] = TIPSS

#################################### one_hot_encoding #######################################
CVA = []
anoxia = []
others = []
#trauma;reference

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('COD_CAD_DON')]
    if a == 2:
        x = 1
    elif np.isnan(a):
        x = 0.5
    elif a == 998:#unknown
        x = 0.5
    else:
        x = 0 
    CVA.append(x)

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('COD_CAD_DON')]
    if a == 1:
        x = 1
    elif np.isnan(a):
        x = np.NAN
    elif a == 998:
        x = np.NAN
    else:
        x = 0 
    anoxia.append(x)

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('COD_CAD_DON')]
    if a == 999:
        x = 1
    elif a == 4:
        x = 1
    elif np.isnan(a):
        x = np.NAN
    elif a == 998:
        x = np.NAN
    else:
        x = 0 
    others.append(x)

df['CVA'] = CVA
df['anoxia'] = anoxia
df['non_CVA/trauma/anoxia'] = others


drug_intoxication =[]
SEIZURE = []

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DEATH_MECH_DON')]
    if a == 1:
        x = 1
    elif np.isnan(a):
        x = np.NAN
    elif a == 998:#unknown
        x = np.NAN
    else:
        x = 0 
    SEIZURE.append(x)

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DEATH_MECH_DON')]
    if a == 3:
        x = 1
    elif np.isnan(a):
        x = np.NAN
    elif a == 998:#unknown
        x = np.NAN
    else:
        x = 0 
    drug_intoxication.append(x)

df["SEIZURE"] = SEIZURE
df["drug_intoxication"] = drug_intoxication


#################################### label encoding #######################################
COD = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('COD_CAD_DON')]
    if a == 1:
        x = 2 #anoxia
    elif a == 2:
        x = 3 #CVA
    elif a == 3:
        x = 0 #trauma
    elif np.isnan(a):
        x = np.NAN
    elif a == 998:#unknown
        x = np.NAN
    elif a == 999:
        x = 1 #others
    elif a == 4:
        x = 1 #others
    COD.append(x)

df['COD'] = COD




#################################### one_hot_encoding #######################################
Partial_split = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TX_PROCEDUR_TY')]
    if a == 702:
        x = 1
    elif a == 703:
        x = 1
    elif np.isnan(a):
        x = np.NAN
    else:
        x = 0 
    Partial_split.append(x)

df['Partial/Split'] = Partial_split


age_don = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('AGE_DON')]
    if 0 <= a < 18:
        x = np.NAN
    elif 18 <= a < 39:
        x = 0
    elif 40 <= a < 49:
        x = 1
    elif 50 <= a < 59:
        x = 2
    elif 60 <= a < 69:
        x = 3
    elif 70 <= a:
        x = 4
    else:
        x = np.NAN #NAN
    age_don.append(x)

df['AGE_DON'] = age_don

GENDER_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('GENDER_DON')]
    if a == 'M':
        x = 0
    elif a == 'F':
        x = 1
    else:
        x = np.NAN
    GENDER_DON.append(x)

df['GENDER_DON'] = GENDER_DON
BMI_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('BMI_DON_CALC')]
    if 0 <= a < 35:
        x = 0
    elif 35 <= a < 40:
        x = 1
    elif 40 <= a:
        x = 2
    else:
        x = np.NAN #NAN
    BMI_DON.append(x)

df['BMI_DON'] = BMI_DON

# HGT = []
# for i in range(len(df)):  
#     a = df.iat[i, df.columns.get_loc('HGT_CM_DON_CALC')]
#     if a <= 170:
#         x = 0
#     elif a > 170:
#         x = 1
#     else:
#         x = np.NAN #NAN
#     HGT.append(x)

# df['HGT_DON'] = HGT

df['HGT_DON']= (df['HGT_CM_DON_CALC']-170)/10


DISTANCE = []#連続で
for i in range(len(df)):  
    a = df.iat[i, df.columns.get_loc('DISTANCE')]
    if a <= 250:
        x = 0
    elif a > 250:
        x = 1
    else:
        x = np.NAN #NAN
    DISTANCE.append(x)

df['DISTANCE'] = DISTANCE

# COLD_ISCHEMIC = []
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('COLD_ISCH')]
#     if 0 <= a < 3:
#         x = 0
#     elif 3 <= a < 6:
#         x = 1
#     elif 6 <= a < 9:
#         x = 2
#     elif 9 <= a < 12:
#         x = 3
#     elif 12 <= a:
#         x = 4 
#     else:
#         x = np.NAN #NAN 
#     COLD_ISCHEMIC.append(x)

# df['COLD_ISCH'] = COLD_ISCHEMIC
p = df['COLD_ISCH'].mean(skipna=True)-1.96*df['COLD_ISCH'].std(skipna=True)
q = df['COLD_ISCH'].mean(skipna=True)+1.96*df['COLD_ISCH'].std(skipna=True)



















COLD_ISCHEMIC = []
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('COLD_ISCH')]
    if a <= p:
        x = p
    elif p < a < q:
        x = a
    elif q <= a:
        x = q
    else:
        x = np.NAN #NAN 
    COLD_ISCHEMIC.append(x)

df['COLD_ISCH'] = COLD_ISCHEMIC


DIABETES_DON = []
HIST_HYPERTENS_DON = []

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIABETES_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    DIABETES_DON.append(x)

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_HYPERTENS_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    HIST_HYPERTENS_DON.append(x)

df['DIABETES_DON'] = DIABETES_DON
df['HIST_HYPERTENS_DON'] = HIST_HYPERTENS_DON

# TB = []
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('TBILI_DON')]
#     if a <= 1.0:
#         x = 0
#     elif 1.0 < a <= 2.0:
#         x = 1
#     elif 2.0 < a <= 3.0:
#         x = 2
#     elif 3.0 < a:
#         x = 3
#     else:
#         x = np.NAN #NAN 
#     TB.append(x)

# df['TBILI_DON'] = TB


# tb = df['TBILI_DON'].mean(skipna=True)+1.96*df['TBILI_DON'].std(skipna=True)
# TBILI_DON = []
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('TBILI_DON')]
#     if a < tb:
#         x = a
#     elif tb <= a:
#         x = tb
#     else:
#         x = np.NAN #NAN 
#     TBILI_DON.append(x)

# df['TBILI_DON'] = TBILI_DON


# SGOT_DON = []#logmodel
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('SGOT_DON')]
#     if a <= 40:
#         x = 0
#     elif 40 < a <= 100:
#         x = 1
#     elif 100 < a <= 200:
#         x = 2
#     elif 200 < a <= 300:
#         x = 3
#     elif 300 < a <= 400:
#         x = 4
#     elif 400 < a <= 500:
#         x = 5
#     elif 500 < a:
#         x = 6
#     else:
#         x = np.NAN #NAN 
#     SGOT_DON.append(x)

# df['SGOT_DON'] = SGOT_DON

SGOT_DON = []#logmodel
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('SGOT_DON')]
    if np.isnan(a):
        x = np.NAN
    elif a == 0:
        x = np.NAN
    else:
        x = np.log10(a)
    SGOT_DON.append(x)
    
df['SGOT_DON'] = SGOT_DON

# SGPT_DON = []#logmodel
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('SGPT_DON')]
#     if a <= 60:
#         x = 0
#     elif 60 < a <= 100:
#         x = 1
#     elif 100 < a <= 200:
#         x = 2
#     elif 200 < a <= 300:
#         x = 3
#     elif 300 < a <= 400:
#         x = 4
#     elif 400 < a <= 500:
#         x = 5
#     elif 500 < a:
#         x = 6
#     else:
#         x = np.NAN #NAN 
#     SGPT_DON.append(x)

# df['SGPT_DON'] = SGPT_DON

SGPT_DON = []#logmodel
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('SGPT_DON')]
    if np.isnan(a):
        x = np.NAN
    elif a == 0:
        x = np.NAN
    else:
        x = np.log10(a+1)
    SGPT_DON.append(x)
    
df['SGPT_DON'] = SGPT_DON

BUN_DON = []#logmodel
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('BUN_DON')]
    if np.isnan(a):
        x = np.NAN
    elif a == 0:
        x = np.NAN
    else:
        x = np.log10(a+1)
    BUN_DON.append(x)
    
df['BUN_DON'] = BUN_DON

CREAT_DON = []#連続で
for i in range(len(df)):  
    a = df.iat[i, df.columns.get_loc('CREAT_DON')]
    if a <= 1.5:
        x = 0
    elif a > 1.5:
        x = 1
    else:
        x = np.NAN #NAN
    CREAT_DON.append(x)

df['CREAT_DON'] = CREAT_DON

# HISTORY_MI_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HISTORY_MI_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = np.NAN
#     else:
#         x = np.NAN
#     HISTORY_MI_DON.append(x)

# df['HISTORY_MI_DON'] = HISTORY_MI_DON

HEMATOCRIT_DON = []#logmodel
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('HEMATOCRIT_DON')]
    if np.isnan(a):
        x = np.NAN
    elif a == 0:
        x = np.NAN
    else:
        x = np.log10(a+1)
    HEMATOCRIT_DON.append(x)
    
df['HEMATOCRIT_DON'] = HEMATOCRIT_DON

# p = df['HEMATOCRIT_DON'].mean(skipna=True)-1.96*df['HEMATOCRIT_DON'].std(skipna=True)
# q = df['HEMATOCRIT_DON'].mean(skipna=True)+1.96*df['HEMATOCRIT_DON'].std(skipna=True)

# HEMATOCRIT_DON = []
# for i in range(len(df)):
#     a = df.iat[i, df.columns.get_loc('HEMATOCRIT_DON')]
#     if a <= p:
#         x = p
#     elif p < a < q:
#         x = a
#     elif q <= a:
#         x = q
#     else:
#         x = np.NAN #NAN 
#     HEMATOCRIT_DON.append(x)

# df['HEMATOCRIT_DON'] = HEMATOCRIT_DON

# CMV_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('CMV_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'P':
#         x = 1
#     elif a == 'U':
#         x = np.NAN
#     elif a == 'ND':
#         x = np.NAN
#     elif a == 'C':
#         x = np.NAN
#     elif a == 'I':
#         x = np.NAN
#     else:
#         x = np.NAN
#     CMV_DON.append(x)

# df['CMV_DON'] = CMV_DON

# ANTIHYPE_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('ANTIHYPE_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     else:
#         x = np.NAN
#     ANTIHYPE_DON.append(x)

# df['ANTIHYPE_DON'] = ANTIHYPE_DON

df['URINE_INF_DON']

# VASODIL_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('VASODIL_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = np.NAN
#     else:
#         x = np.NAN
#     VASODIL_DON.append(x)

# df['VASODIL_DON'] = VASODIL_DON

# VDRL_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('VDRL_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'P':
#         x = 1
#     elif a == 'U':
#         x = np.NAN
#     elif a == 'ND':
#         x = np.NAN
#     else:
#         x = np.NAN
#     VDRL_DON.append(x)

# df['VDRL_DON'] = VDRL_DON

HIST_COCAINE_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_COCAINE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    HIST_COCAINE_DON.append(x)

df['HIST_COCAINE_DON'] = HIST_COCAINE_DON

HIST_CANCER_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CANCER_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    HIST_CANCER_DON.append(x)

df['HIST_CANCER_DON'] = HIST_CANCER_DON

ALCOHOL_HEAVY_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ALCOHOL_HEAVY_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    ALCOHOL_HEAVY_DON.append(x)

df['ALCOHOL_HEAVY_DON'] = ALCOHOL_HEAVY_DON

HIST_CIG_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CIG_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    HIST_CIG_DON.append(x)

df['HIST_CIG_DON'] = HIST_CIG_DON

# HIST_OTH_DRUG_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HIST_OTH_DRUG_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0
#     else:
#         x = np.NAN
#     HIST_OTH_DRUG_DON.append(x)

# df['HIST_OTH_DRUG_DON'] = HIST_OTH_DRUG_DON

# MACRO_FAT_LI_DON = []#連続で
# for i in range(len(df)):  
#     a = df.iat[i, df.columns.get_loc('MACRO_FAT_LI_DON')]
#     if a <= 10:
#         x = 0
#     elif a > 10:
#         x = 1
#     else:
#         x = np.NAN
#     MACRO_FAT_LI_DON.append(x)

# df['MACRO_FAT_LI_DON'] = MACRO_FAT_LI_DON


# MICRO_FAT_LI_DON = []#連続で
# for i in range(len(df)):  
#     a = df.iat[i, df.columns.get_loc('MICRO_FAT_LI_DON')]
#     if a <= 10:
#         x = 0
#     elif a > 10:
#         x = 1
#     else:
#         x = np.NAN
#     MICRO_FAT_LI_DON.append(x)

# df['MICRO_FAT_LI_DON'] = MICRO_FAT_LI_DON

MACRO_FAT_LI_DON = []#連続で
for i in range(len(df)):  
    a = df.iat[i, df.columns.get_loc('MACRO_FAT_LI_DON')]
    if 0 < a <= 30:
        x = 1
    elif 30 < a <= 60:
        x = 2
    elif 60 < a:
        x = 3
    else:
        x = 0
    MACRO_FAT_LI_DON.append(x)

df['MACRO_FAT_LI_DON'] = MACRO_FAT_LI_DON


MICRO_FAT_LI_DON = []#連続で
for i in range(len(df)):  
    a = df.iat[i, df.columns.get_loc('MICRO_FAT_LI_DON')]
    if 0 < a <= 30:
        x = 1
    elif 30 < a <= 60:
        x = 2
    elif 60 < a:
        x = 3
    else:
        x = 0
    MICRO_FAT_LI_DON.append(x)

df['MICRO_FAT_LI_DON'] = MICRO_FAT_LI_DON

NON_HRT_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('NON_HRT_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    else:
        x = np.NAN
    NON_HRT_DON.append(x)

df['NON_HRT_DON'] = NON_HRT_DON 

HEP_C_ANTI_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEP_C_ANTI_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 1
    elif a == 'U':
        x = 0.5
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HEP_C_ANTI_DON.append(x)

df['HEP_C_ANTI_DON'] = HEP_C_ANTI_DON

HBV_SUR_ANTIGEN_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HBV_SUR_ANTIGEN_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 1
    elif a == 'U':
        x = 0.5
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HBV_SUR_ANTIGEN_DON.append(x)

df['HBV_SUR_ANTIGEN_DON'] = HBV_SUR_ANTIGEN_DON

ARGININE_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ARGININE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    ARGININE_DON.append(x)

df['ARGININE_DON'] = ARGININE_DON


HEPARIN_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEPARIN_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0.5
    else:
        x = np.NAN
    HEPARIN_DON.append(x)

df['HEPARIN_DON'] = HEPARIN_DON

ACIDOSIS = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PH_DON')]
    if a < 7.35:
        x = 1
    elif a >= 7.35:
        x = 0
    else:
        x = np.NAN
    ACIDOSIS.append(x)

df['ACIDOSIS'] = ACIDOSIS

ALKALOSIS = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PH_DON')]
    if a > 7.45:
        x = 1
    elif a <= 7.45:
        x = 0
    else:
        x = np.NAN
    ALKALOSIS.append(x)

df['ALKALOSIS'] = ALKALOSIS

TRANSFUS_TERM_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TRANSFUS_TERM_DON')]
    if a == 998:
        x = 0
    elif np.isnan(a):
        x = 0
    else:
        x = a
    TRANSFUS_TERM_DON.append(x)

df['TRANSFUS_TERM_DON'] = TRANSFUS_TERM_DON


CORONARY_ANGIO_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('CORONARY_ANGIO_DON')]
    if a == 'Y':
        x = 1
    else:
        x = 0
    CORONARY_ANGIO_DON.append(x)

df['CORONARY_ANGIO_DON'] = CORONARY_ANGIO_DON

m_to_f = []
f_to_m = []

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('GENDER_DON')]
    b = df.iat[i, df.columns.get_loc('GENDER')]
    if a == 0 and b == 1:
        x = 1
    else:
        x = 0
    m_to_f.append(x)

df['m_to_f'] = m_to_f

for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('GENDER_DON')]
    b = df.iat[i, df.columns.get_loc('GENDER')]
    if a == 1 and b == 0:
        x = 1
    else:
        x = 0
    f_to_m.append(x)

df['f_to_m'] = f_to_m


GTIME_1m = []
GSTATUS_1m = []
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('GTIME')]
    b = df.iat[i, df.columns.get_loc('GSTATUS')]
    if a > 30:
        x = 30
        y = 0
    else:
        x = a 
        y = b
    GTIME_1m.append(x)
    GSTATUS_1m.append(y)

df['GTIME_1m'] = GTIME_1m
df['GSTATUS_1m'] = GSTATUS_1m

GTIME_3m = []
GSTATUS_3m = []
for i in range(len(df)):
    a = df.iat[i, df.columns.get_loc('GTIME')]
    b = df.iat[i, df.columns.get_loc('GSTATUS')]
    if a > 14:
        x = 14
        y = 0
    else:
        x = a 
        y = b
    GTIME_3m.append(x)
    GSTATUS_3m.append(y)

df['GTIME_3m'] = GTIME_3m
df['GSTATUS_3m'] = GSTATUS_3m

df1 = df[['GSTATUS_3m',
          'TX_YEAR',
         'NON_HRT_DON',
         'CVA','anoxia',
         'non_CVA/trauma/anoxia',
        #  'SEIZURE',
        #  'drug_intoxication',
         'Partial/Split',
         'AGE_DON',
         'GENDER_DON',
         'BMI_DON',
         'HGT_DON',
         'DISTANCE',
         'COLD_ISCH',
         'DIABETES_DON',
         'HIST_HYPERTENS_DON',
         'TBILI_DON',
         'SGOT_DON',
         'SGPT_DON',
         'BUN_DON',
         'CREAT_DON',
         'BLOOD_INF_DON',
         'MACRO_FAT_LI_DON',
         'MICRO_FAT_LI_DON',
        #  'A&BLocus',
        #  'DRLocus',
        #  'DA1',
        #  'DA2',
        #  'DB1',
        #  'DB2',
        #  'DDR1',
        #  'DDR2',
         'HEMATOCRIT_DON',
         'URINE_INF_DON',
         'HIST_COCAINE_DON',
         'HIST_CANCER_DON',
         'ALCOHOL_HEAVY_DON',
         'HIST_CIG_DON',
         'PH_DON',
        #  'ACIDOSIS',
        #  'ALKALOSIS',
         'HEP_C_ANTI_DON',
         'HBV_SUR_ANTIGEN_DON',
         'HEPARIN_DON',
        #  'ARGININE_DON'
        'TRANSFUS_TERM_DON',
        'CORONARY_ANGIO_DON',
        # 'f_to_m',
        # 'm_to_f',
        ####recipient####
        'AGE',
        'GENDER',
        'MED_COND_TRR',
        'PREV_TX',
        'BMI_REC',
        # 'MELD_NA',
        'ASCITES_TX',
        'ALBUMIN_TX',
        'TBILI_TX',#meldとは一緒にしないもしくはmeldを抜く
        'CREAT_TX',
        'DIAB',
        'HGT_CM_CALC',
        'AHN',
        'AIH_PBC_PSC',
        'HCV',#diagnosis
        'MALIG',
        'Cholestatic_disease',
        'PORTAL_VEIN_TRR',
        'PREV_AB_SURG_TCR',
        'ON_VENT_TRR',
        'PROTEIN_URINE',
        'CARDARREST_NEURO',
        'TATTOOS',
        'TIPSS_TRR'
         ]]
df1 = df1.dropna(axis=0, how='any')

import optuna
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb
import tqdm as tqdm


###############################Rolling_XGboost#################################
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
def XGBOptuna(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import optuna
    # 乱数の選択範囲は1〜1500とした
    random_state = trial.suggest_int('random_state', 1, 1500)#元々1500
    import xgboost as xgb
    model_xgb = xgb.XGBRegressor(objective = 'binary:logistic', 
                                 colsample_bytree = 0.3, 
                                 learning_rate = 0.3, 
                                 max_depth = 7, 
                                 alpha = 10, 
                                 n_etimator= 10,
                                 random_state = random_state,#これ忘れてた。。。
                                 verbose = 0,
                                 verbosity = 0)
    model_xgb.fit(x_train, t_train)
    predicted_xgb = model_xgb.predict(x_test)
    auc = roc_auc_score(t_test, predicted_xgb)
    # モデルの保存
    # import pickle
    # directory = '/Users/yanagawarintaro/Desktop/research/re_DRI_study/model'
    # with open(directory + '/model' + str(i) + 'lgmopt.pickle', mode = 'wb') as f:
    #     pickle.dump(model_light, f)
    return 1/auc

def XGBTrial(n_trials):
    import optuna
    study = optuna.create_study()
    study.optimize(XGBOptuna, n_trials)
    # print('ハイパーパラメーター: ', study.best_params)
    # print('AUC: ', 1/study.best_value)

N = 3
df_roc_total_N = pd.DataFrame(index=[i/1000 for i in range(1001)])
auc_T = []##########
for p in tqdm.tqdm(range(N)):
    YEAR = list(set(df1['TX_YEAR'].tolist()))
    YEAR.sort()
    YEAR.remove(2010)
    auc_toal = []
    auc = []########
    df_roc_total = pd.DataFrame(index=[i/1000 for i in range(1001)])
    for i in tqdm.tqdm(YEAR):
        auc_i_list = []
        df_roc_total_k = pd.DataFrame(index=[i/1000 for i in range(1001)])
        for k in tqdm.tqdm(range(i-2010)):
            l_k = [j+i-k-1 for j in range(k+1)]
            n=10
            k_auc_list = []
            df_roc0 = pd.DataFrame(index=[i/1000 for i in range(1001)])
            for v in tqdm.tqdm(range(n)):#down sampling
                df_train = df1[df1['TX_YEAR'].isin(l_k)]
                df_train1 = df_train[df_train['GSTATUS_3m'] == 1]
                df_train0 = df_train[df_train['GSTATUS_3m'] == 0].sample(n=len(df_train1))
                DF_train = pd.concat([df_train1, df_train0])
                x_train = DF_train.drop('GSTATUS_3m', axis=1)
                x_train = x_train.drop('TX_YEAR', axis=1)
                t_train = np.array(DF_train['GSTATUS_3m'].tolist())
                df_test = df1[df1['TX_YEAR'] == i]
                df_test1 = df_test[df_test['GSTATUS_3m'] == 1]
                df_test0 = df_test[df_test['GSTATUS_3m'] == 0].sample(n=len(df_test1))
                DF_test = pd.concat([df_test1, df_test0])
                x_test = DF_test.drop('GSTATUS_3m', axis=1)
                x_test = x_test.drop('TX_YEAR', axis=1)
                t_test = np.array(DF_test['GSTATUS_3m'].tolist())
                from sklearn.linear_model import LogisticRegression
                model_l = LogisticRegression(verbose=0)
                model_l.fit(x_train, t_train)
                predicted_l = model_l.predict(x_test)
                proba_l = model_l.predict_proba(x_test)[:, 1]
                k_auc_list.append(roc_auc_score(t_test, proba_l))
                # k_auc_list.append(1/study.best_value)
                # model_xgb = xgb.XGBRegressor(objective = 'binary:logistic', 
                #                             colsample_bytree = 0.3, 
                #                             learning_rate = 0.3, max_depth = 7, alpha = 10, n_etimator= 10, 
                #                             random_state = study.best_params['random_state'],
                #                             verbosity = 0)
                # model_xgb.fit(x_train, t_train)
                # predicted_xgb = model_xgb.predict(x_test)
                # from sklearn.metrics import roc_curve, roc_auc_score
                # fpr, tpr, thresholds = roc_curve(t_test, predicted_xgb)
                # df_roc = pd.DataFrame(
                # data={'fpr': list(np.round(fpr.tolist(),3)), 
                #     'tpr': tpr.tolist()}
                # )
                # fpr_list = list(set(list(np.round(fpr.tolist(),3))))
                # fpr_list.sort()
                # tpr_list = []
                # for u in fpr_list:
                #     tpr_list.append(df_roc[df_roc['fpr'] == u]['tpr'].mean())
                # df_roc2 = pd.DataFrame(tpr_list, index=fpr_list)
                # df_roc2 = df_roc2.reindex([i/1000 for i in range(1001)])
                # df_roc0[v] = df_roc2.interpolate(limit_direction='both')[0].tolist()
            auc_k = [round(np.mean(k_auc_list),3), '('+ str(round(np.mean(k_auc_list)-1.96*stdev(k_auc_list),3)) + '-' + str(round(np.mean(k_auc_list)+1.96*stdev(k_auc_list),3))+')']
            auc_i_list.append(auc_k)
            auc.append(np.mean(k_auc_list))#############
            # df_roc_total_k[k] = df_roc0.mean(axis='columns').tolist() 
        #この線はiのライン
        auc_toal.append(auc_i_list +[np.NAN]*(max(YEAR)-i))
        # df_roc_total[i] = df_roc_total_k.mean(axis='columns').tolist()
    #この線はpのライン
    #ここでaucが出来上がっている
    auc_T.append(np.mean(auc))#########
    # df_roc_total_N[p] = df_roc_total.mean(axis='columns').tolist()





[round(np.mean(auc_T),3), '('+ str(round(np.mean(auc_T)-1.96*stdev(auc_T),3)) + '-' + str(round(np.mean(auc_T)+1.96*stdev(auc_T),3))+')']
# tpr_rolling_xgb = np.array(df_roc_total_N.mean(axis='columns').tolist())









# # df_train = df1[df1['TX_YEAR'] == 1]
# YEAR = list(set(df1['TX_YEAR'].tolist()))
# YEAR.sort()
# YEAR.remove(2010)
# auc_toal = []
# auc = []
# df_roc_total = pd.DataFrame(index=[i/1000 for i in range(1001)])
# for i in YEAR:
#     auc_i_list = []
#     df_roc_total_k = pd.DataFrame(index=[i/1000 for i in range(1001)])
#     for k in range(i-2010):
#         l_k = [j+i-k-1 for j in range(k+1)]
#         n=10
#         k_auc_list = []
#         df_roc0 = pd.DataFrame(index=[i/1000 for i in range(1001)])
#         for v in range(n):#down sampling
#             df_train = df1[df1['TX_YEAR'].isin(l_k)]
#             df_train1 = df_train[df_train['GSTATUS_3m'] == 1]
#             df_train0 = df_train[df_train['GSTATUS_3m'] == 0].sample(n=len(df_train1))
#             DF_train = pd.concat([df_train1, df_train0])
#             x_train = DF_train.drop('GSTATUS_3m', axis=1)
#             x_train = x_train.drop('TX_YEAR', axis=1)
#             t_train = np.array(DF_train['GSTATUS_3m'].tolist())
#             df_test = df1[df1['TX_YEAR'] == i]
#             df_test1 = df_test[df_test['GSTATUS_3m'] == 1]
#             df_test0 = df_test[df_test['GSTATUS_3m'] == 0].sample(n=len(df_test1))
#             DF_test = pd.concat([df_test1, df_test0])
#             x_test = DF_test.drop('GSTATUS_3m', axis=1)
#             x_test = x_test.drop('TX_YEAR', axis=1)
#             t_test = np.array(DF_test['GSTATUS_3m'].tolist())
#             # import xgboost as xgb
#             # model_xgb = xgb.XGBRegressor(objective = 'binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3, max_depth = 7, alpha = 10, n_etimator= 10, 
#             #                              verbosity = 0)
#             # model_xgb.fit(x_train, t_train)
#             # predicted_xgb = model_xgb.predict(x_test)
#             # k_auc_list.append(roc_auc_score(t_test, predicted_xgb))
#             # auc_xgb = 0
#             # import time
#             # timeout = time.time() + 60*10
#             # auc_max = []
#             # while auc_xgb <= 0.63:
#             #     study = optuna.create_study()
#             #     optuna.logging.set_verbosity(optuna.logging.WARNING)
#             #     study.optimize(XGBOptuna, 10)#trial数10の方が良い結果が出てくる？
#             #     auc_xgb = 1/study.best_value
#             #     auc_max.append(auc_xgb)
#             #     if time.time() > timeout:
#             #         break
#             #     k_auc_list.append(max(auc_max))
#             study = optuna.create_study()
#             optuna.logging.set_verbosity(optuna.logging.WARNING)
#             study.optimize(XGBOptuna, 10)#trial数10の方が良い結果が出てくる？
#             k_auc_list.append(1/study.best_value)
#             model_xgb = xgb.XGBRegressor(objective = 'binary:logistic', 
#                                          colsample_bytree = 0.3, 
#                                          learning_rate = 0.3, max_depth = 7, alpha = 10, n_etimator= 10, 
#                                          random_state = study.best_params['random_state'],
#                                          verbosity = 0)
#             model_xgb.fit(x_train, t_train)
#             predicted_xgb = model_xgb.predict(x_test)
#             from sklearn.metrics import roc_curve, roc_auc_score
#             fpr, tpr, thresholds = roc_curve(t_test, predicted_xgb)
#             df_roc = pd.DataFrame(
#             data={'fpr': list(np.round(fpr.tolist(),3)), 
#                 'tpr': tpr.tolist()}
#             )
#             fpr_list = list(set(list(np.round(fpr.tolist(),3))))
#             fpr_list.sort()
#             tpr_list = []
#             for u in fpr_list:
#                 tpr_list.append(df_roc[df_roc['fpr'] == u]['tpr'].mean())
#             df_roc2 = pd.DataFrame(tpr_list, index=fpr_list)
#             df_roc2 = df_roc2.reindex([i/1000 for i in range(1001)])
#             df_roc0[v] = df_roc2.interpolate(limit_direction='both')[0].tolist()
#         auc_k = [round(np.mean(k_auc_list),3), '('+ str(round(np.mean(k_auc_list)-1.96*stdev(k_auc_list),3)) + '-' + str(round(np.mean(k_auc_list)+1.96*stdev(k_auc_list),3))+')']
#         auc_i_list.append(auc_k)
#         auc.append(np.mean(k_auc_list))
#         df_roc_total_k[k] = df_roc0.mean(axis='columns').tolist()  
#     auc_toal.append(auc_i_list +[np.NAN]*(max(YEAR)-i))
#     df_roc_total[i] = df_roc_total_k.mean(axis='columns').tolist()

# rolling_xgb = pd.DataFrame(
#     index=[j for j in range(10)],
#     columns=[j + 2010 +1 for j in range(10)]
#              )
# for j in range(10):
#     auc_toal[j]
#     rolling_xgb[2010+1+j] = auc_toal[j]

# rolling_xgb
# tpr_rolling_xgb = np.array(df_roc_total.mean(axis='columns').tolist())
# [round(np.mean(auc),3), '('+ str(round(np.mean(auc)-1.96*stdev(auc),3)) + '-' + str(round(np.mean(auc)+1.96*stdev(auc),3))+')']

# rolling_xgb_90days_roc = pd.DataFrame(
#     index=[i for i in range(1001)],
#              )

# rolling_xgb_90days_roc['fpr'] = [i/1000 for i in range(1001)]
# rolling_xgb_90days_roc['tpr'] = tpr_rolling_xgb

# rolling_xgb_90days_roc.to_excel('rolling_CV_xgb_14days.xlsx')


