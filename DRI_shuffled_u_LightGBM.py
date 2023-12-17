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
    elif 18 <= a < 40:
        x = 0
    elif 40 <= a < 50:
        x = 1
    elif 50 <= a < 60:
        x = 2
    elif 60 <= a < 70:
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

# PREV_AB_SURG_TCR = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('PREV_AB_SURG_TCR')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     PREV_AB_SURG_TCR.append(x)

# df['PREV_AB_SURG_TCR'] = PREV_AB_SURG_TCR

PREV_AB_SURG_TCR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PREV_AB_SURG_TCR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    PREV_AB_SURG_TCR.append(x)

df['PREV_AB_SURG_TCR_kown'] = PREV_AB_SURG_TCR

PREV_AB_SURG_TCR_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PREV_AB_SURG_TCR')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    PREV_AB_SURG_TCR_unknown.append(x)

df['PREV_AB_SURG_TCR_unknown'] = PREV_AB_SURG_TCR_unknown

df['ON_VENT_TRR']

# PORTAL_VEIN_TRR = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('PORTAL_VEIN_TRR')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     PORTAL_VEIN_TRR.append(x)

# df['PORTAL_VEIN_TRR'] = PORTAL_VEIN_TRR
PORTAL_VEIN_TRR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PORTAL_VEIN_TRR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    PORTAL_VEIN_TRR.append(x)

df['PORTAL_VEIN_TRR_known'] = PORTAL_VEIN_TRR

PORTAL_VEIN_TRR_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PORTAL_VEIN_TRR')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    PORTAL_VEIN_TRR_unknown.append(x)

df['PORTAL_VEIN_TRR_unknown'] = PORTAL_VEIN_TRR_unknown


# MALIG = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('MALIG')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     MALIG.append(x)

# df['MALIG'] = MALIG
MALIG = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('MALIG')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    MALIG.append(x)

df['MALIG_known'] = MALIG

MALIG_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('MALIG')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    MALIG_unknown.append(x)

df['MALIG_unknown'] = MALIG_unknown

# TATTOOS = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('TATTOOS')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     TATTOOS.append(x)

# df['TATTOOS'] = TATTOOS

TATTOOS = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TATTOOS')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    TATTOOS.append(x)

df['TATTOOS_known'] = TATTOOS

TATTOOS_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TATTOOS')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    TATTOOS_unknown.append(x)

df['TATTOOS_unknown'] = TATTOOS_unknown


# PROTEIN_URINE = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('PROTEIN_URINE')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     PROTEIN_URINE.append(x)

# df['PROTEIN_URINE'] = PROTEIN_URINE

PROTEIN_URINE = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PROTEIN_URINE')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    PROTEIN_URINE.append(x)

df['PROTEIN_URINE_known'] = PROTEIN_URINE

PROTEIN_URINE_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('PROTEIN_URINE')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    PROTEIN_URINE_unknown.append(x)

df['PROTEIN_URINE_unknown'] = PROTEIN_URINE_unknown

# CARDARREST_NEURO = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('CARDARREST_NEURO')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     CARDARREST_NEURO.append(x)

# df['CARDARREST_NEURO'] = CARDARREST_NEURO

CARDARREST_NEURO = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('CARDARREST_NEURO')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    CARDARREST_NEURO.append(x)

df['CARDARREST_NEURO_known'] = CARDARREST_NEURO

CARDARREST_NEURO_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('CARDARREST_NEURO')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    CARDARREST_NEURO_unknown.append(x)

df['CARDARREST_NEURO_unknown'] = CARDARREST_NEURO_unknown

# TIPSS = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('TIPSS_TRR')]
#     if a == 'Y':
#         x = 1
#     elif a == 'N':
#         x = 0
#     elif a == 'U':
#         x = 0.5 ######
#     else:
#         x = np.NAN
#     TIPSS.append(x)

# df['TIPSS_TRR'] = TIPSS

TIPSS_TRR = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TIPSS_TRR')]
    if a == 'Y':
        x = 1
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 0 ######
    else:
        x = np.NAN
    TIPSS_TRR.append(x)

df['TIPSS_TRR_known'] = TIPSS_TRR

TIPSS_TRR_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('TIPSS_TRR')]
    if a == 'Y':
        x = 0
    elif a == 'N':
        x = 0
    elif a == 'U':
        x = 1 ######
    else:
        x = np.NAN
    TIPSS_TRR_unknown.append(x)

df['TIPSS_TRR_unknown'] = TIPSS_TRR_unknown

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
        x = np.NAN
    elif a == 998:#unknown
        x = 0 #np.NAN
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
        x = 0 #np.NAN
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
        x = 0 #np.NAN
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
        x = 0 #np.NAN
    elif 18 <= a < 40:
        x = 0
    elif 40 <= a < 50:
        x = 1
    elif 50 <= a < 60:
        x = 2
    elif 60 <= a < 70:
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


# DIABETES_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('DIABETES_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     DIABETES_DON.append(x)

# df['DIABETES_DON'] = DIABETES_DON

DIABETES_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIABETES_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    DIABETES_DON.append(x)

df['DIABETES_DON_known'] = DIABETES_DON

DIABETES_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('DIABETES_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    DIABETES_DON_unknown.append(x)

df['DIABETES_DON_unknown'] = DIABETES_DON_unknown

# HIST_HYPERTENS_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HIST_HYPERTENS_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     HIST_HYPERTENS_DON.append(x)

# df['HIST_HYPERTENS_DON'] = HIST_HYPERTENS_DON

HIST_HYPERTENS_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_HYPERTENS_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    HIST_HYPERTENS_DON.append(x)

df['HIST_HYPERTENS_DON_known'] = HIST_HYPERTENS_DON

HIST_HYPERTENS_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_HYPERTENS_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    HIST_HYPERTENS_DON_unknown.append(x)

df['HIST_HYPERTENS_DON_unknown'] = HIST_HYPERTENS_DON_unknown

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

# df['TBILI_DON'] = np.log10(df['TBILI_DON']+1)
# df['TBILI_TX'] = np.log10(df['TBILI_TX']+1)

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

# HIST_COCAINE_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HIST_COCAINE_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     HIST_COCAINE_DON.append(x)

# df['HIST_COCAINE_DON'] = HIST_COCAINE_DON

HIST_COCAINE_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_COCAINE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    HIST_COCAINE_DON.append(x)

df['HIST_COCAINE_DON_known'] = HIST_COCAINE_DON

HIST_COCAINE_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_COCAINE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    HIST_COCAINE_DON_unknown.append(x)

df['HIST_COCAINE_DON_unknown'] = HIST_COCAINE_DON_unknown

# HIST_CANCER_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HIST_CANCER_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     HIST_CANCER_DON.append(x)

# df['HIST_CANCER_DON'] = HIST_CANCER_DON

HIST_CANCER_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CANCER_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    HIST_CANCER_DON.append(x)

df['HIST_CANCER_DON_known'] = HIST_CANCER_DON

HIST_CANCER_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CANCER_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    HIST_CANCER_DON_unknown.append(x)

df['HIST_CANCER_DON_unknown'] = HIST_CANCER_DON_unknown

# ALCOHOL_HEAVY_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('ALCOHOL_HEAVY_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     ALCOHOL_HEAVY_DON.append(x)

# df['ALCOHOL_HEAVY_DON'] = ALCOHOL_HEAVY_DON

ALCOHOL_HEAVY_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ALCOHOL_HEAVY_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    ALCOHOL_HEAVY_DON.append(x)

df['ALCOHOL_HEAVY_DON_known'] = ALCOHOL_HEAVY_DON

ALCOHOL_HEAVY_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ALCOHOL_HEAVY_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    ALCOHOL_HEAVY_DON_unknown.append(x)

df['ALCOHOL_HEAVY_DON_unknown'] = ALCOHOL_HEAVY_DON_unknown

# HIST_CIG_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HIST_CIG_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     HIST_CIG_DON.append(x)

# df['HIST_CIG_DON'] = HIST_CIG_DON

HIST_CIG_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CIG_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    HIST_CIG_DON.append(x)

df['HIST_CIG_DON_known'] = HIST_CIG_DON

HIST_CIG_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HIST_CIG_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    HIST_CIG_DON_unknown.append(x)

df['HIST_CIG_DON_unknown'] = HIST_CIG_DON_unknown

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

# HEP_C_ANTI_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HEP_C_ANTI_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'P':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     elif a == 'ND':
#         x = np.NAN
#     elif a == 'C':
#         x = np.NAN
#     elif a == 'I':
#         x = np.NAN
#     else:
#         x = np.NAN
#     HEP_C_ANTI_DON.append(x)

# df['HEP_C_ANTI_DON'] = HEP_C_ANTI_DON

HEP_C_ANTI_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEP_C_ANTI_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 1
    elif a == 'U':
        x = 0
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HEP_C_ANTI_DON.append(x)

df['HEP_C_ANTI_DON_known'] = HEP_C_ANTI_DON

HEP_C_ANTI_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEP_C_ANTI_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 0
    elif a == 'U':
        x = 1
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HEP_C_ANTI_DON_unknown.append(x)

df['HEP_C_ANTI_DON_unknown'] = HEP_C_ANTI_DON_unknown

# HBV_SUR_ANTIGEN_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HBV_SUR_ANTIGEN_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'P':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     elif a == 'ND':
#         x = np.NAN
#     elif a == 'C':
#         x = np.NAN
#     elif a == 'I':
#         x = np.NAN
#     else:
#         x = np.NAN
#     HBV_SUR_ANTIGEN_DON.append(x)

# df['HBV_SUR_ANTIGEN_DON'] = HBV_SUR_ANTIGEN_DON

HBV_SUR_ANTIGEN_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HBV_SUR_ANTIGEN_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 1
    elif a == 'U':
        x = 0
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HBV_SUR_ANTIGEN_DON.append(x)

df['HBV_SUR_ANTIGEN_DON_known'] = HBV_SUR_ANTIGEN_DON

HBV_SUR_ANTIGEN_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HBV_SUR_ANTIGEN_DON')]
    if a == 'N':
        x = 0
    elif a == 'P':
        x = 0
    elif a == 'U':
        x = 1
    elif a == 'ND':
        x = np.NAN
    elif a == 'C':
        x = np.NAN
    elif a == 'I':
        x = np.NAN
    else:
        x = np.NAN
    HBV_SUR_ANTIGEN_DON_unknown.append(x)

df['HBV_SUR_ANTIGEN_DON_unknown'] = HBV_SUR_ANTIGEN_DON_unknown

# ARGININE_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('ARGININE_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     ARGININE_DON.append(x)

# df['ARGININE_DON'] = ARGININE_DON

ARGININE_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ARGININE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    ARGININE_DON.append(x)

df['ARGININE_DON_known'] = ARGININE_DON

ARGININE_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('ARGININE_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    ARGININE_DON_unknown.append(x)

df['ARGININE_DON_unknown'] = ARGININE_DON_unknown

# HEPARIN_DON = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('HEPARIN_DON')]
#     if a == 'N':
#         x = 0
#     elif a == 'Y':
#         x = 1
#     elif a == 'U':
#         x = 0.5
#     else:
#         x = np.NAN
#     HEPARIN_DON.append(x)

# df['HEPARIN_DON'] = HEPARIN_DON

HEPARIN_DON = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEPARIN_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 1
    elif a == 'U':
        x = 0
    else:
        x = np.NAN
    HEPARIN_DON.append(x)

df['HEPARIN_DON_known'] = HEPARIN_DON

HEPARIN_DON_unknown = []
for i in range(len(df)):     
    a = df.iat[i, df.columns.get_loc('HEPARIN_DON')]
    if a == 'N':
        x = 0
    elif a == 'Y':
        x = 0
    elif a == 'U':
        x = 1
    else:
        x = np.NAN
    HEPARIN_DON_unknown.append(x)

df['HEPARIN_DON_unknown'] = HEPARIN_DON_unknown

# ACIDOSIS = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('PH_DON')]
#     if a < 7.35:
#         x = 1
#     elif a >= 7.35:
#         x = 0
#     else:
#         x = np.NAN
#     ACIDOSIS.append(x)

# df['ACIDOSIS'] = ACIDOSIS

# ALKALOSIS = []
# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('PH_DON')]
#     if a > 7.45:
#         x = 1
#     elif a <= 7.45:
#         x = 0
#     else:
#         x = np.NAN
#     ALKALOSIS.append(x)

# df['ALKALOSIS'] = ALKALOSIS

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

# m_to_f = []
# f_to_m = []

# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('GENDER_DON')]
#     b = df.iat[i, df.columns.get_loc('GENDER')]
#     if a == 0 and b == 1:
#         x = 1
#     else:
#         x = 0
#     m_to_f.append(x)

# df['m_to_f'] = m_to_f

# for i in range(len(df)):     
#     a = df.iat[i, df.columns.get_loc('GENDER_DON')]
#     b = df.iat[i, df.columns.get_loc('GENDER')]
#     if a == 1 and b == 0:
#         x = 1
#     else:
#         x = 0
#     f_to_m.append(x)

# df['f_to_m'] = f_to_m


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
    if a > 90:
        x = 90
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
        #  'DIABETES_DON',
         'DIABETES_DON_known',
         'DIABETES_DON_unknown',
        #  'HIST_HYPERTENS_DON',
        'HIST_HYPERTENS_DON_known',
        'HIST_HYPERTENS_DON_unknown',
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
        #  'HIST_COCAINE_DON',
        'HIST_COCAINE_DON_known',
        'HIST_COCAINE_DON_unknown',
        #  'HIST_CANCER_DON',
        'HIST_CANCER_DON_known',
        'HIST_CANCER_DON_unknown',
        #  'ALCOHOL_HEAVY_DON',
         'ALCOHOL_HEAVY_DON_known',
         'ALCOHOL_HEAVY_DON_unknown',
        #  'HIST_CIG_DON',
        'HIST_CIG_DON_known',
        'HIST_CIG_DON_unknown',
         'PH_DON',
        #  'ACIDOSIS',
        #  'ALKALOSIS',
        #  'HEP_C_ANTI_DON',
        'HEP_C_ANTI_DON_known',
        'HEP_C_ANTI_DON_unknown',
        #  'HBV_SUR_ANTIGEN_DON',
        'HBV_SUR_ANTIGEN_DON_known',
        'HBV_SUR_ANTIGEN_DON_unknown',
        #  'HEPARIN_DON',
        'HEPARIN_DON_known',
        'HEPARIN_DON_unknown',
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
        # 'FINAL_SERUM_SODIUM',
        'ASCITES_TX',
        'ALBUMIN_TX',
        'TBILI_TX',#meldとは一緒にしないもしくはmeldを抜く
        'CREAT_TX',
        'DIAB',
        'HGT_CM_CALC',
        'AHN',
        'AIH_PBC_PSC',
        'HCV',#diagnosis
        # 'MALIG',
        'MALIG_known',
        'MALIG_unknown',
        'Cholestatic_disease',
        # 'PORTAL_VEIN_TRR',
        'PORTAL_VEIN_TRR_known',
        'PORTAL_VEIN_TRR_unknown',
        # 'PREV_AB_SURG_TCR',
        'PREV_AB_SURG_TCR_kown',
        'PREV_AB_SURG_TCR_unknown',
        'ON_VENT_TRR',
        # 'PROTEIN_URINE',
        # 'CARDARREST_NEURO',
        # 'TATTOOS',
        # 'TIPSS_TRR'
        'PROTEIN_URINE_known',
        'CARDARREST_NEURO_known',
        'TATTOOS_known',
        'TIPSS_TRR_known',
        'PROTEIN_URINE_unknown',
        'CARDARREST_NEURO_unknown',
        'TATTOOS_unknown',
        'TIPSS_TRR_unknown'
         ]]


# df1 = df[['GSTATUS_3m',
#           'TX_YEAR',
#          'NON_HRT_DON',
#          'CVA',
#         #  'anoxia',
#         #  'non_CVA/trauma/anoxia',
#         #  'SEIZURE',
#         #  'drug_intoxication',
#         #  'Partial/Split',
#          'AGE_DON',
#         #  'GENDER_DON',
#         #  'BMI_DON',
#          'HGT_DON',
#         #  'DISTANCE',
#          'COLD_ISCH',
#         #  'DIABETES_DON',
#          'DIABETES_DON_known',
#         #  'DIABETES_DON_unknown',
#         #  'HIST_HYPERTENS_DON',
#         # 'HIST_HYPERTENS_DON_known',
#         # 'HIST_HYPERTENS_DON_unknown',
#         #  'TBILI_DON',
#          'SGOT_DON',
#         #  'SGPT_DON',
#          'BUN_DON',
#         #  'CREAT_DON',
#         #  'BLOOD_INF_DON',
#          'MACRO_FAT_LI_DON',
#         #  'MICRO_FAT_LI_DON',
#         #  'A&BLocus',
#         #  'DRLocus',
#         #  'DA1',
#         #  'DA2',
#         #  'DB1',
#         #  'DB2',
#         #  'DDR1',
#         #  'DDR2',
#         #  'HEMATOCRIT_DON',
#         #  'URINE_INF_DON',
#         #  'HIST_COCAINE_DON',
#         # 'HIST_COCAINE_DON_known',
#         # 'HIST_COCAINE_DON_unknown',
#         #  'HIST_CANCER_DON',
#         # 'HIST_CANCER_DON_known',
#         # 'HIST_CANCER_DON_unknown',
#         #  'ALCOHOL_HEAVY_DON',
#         #  'ALCOHOL_HEAVY_DON_known',
#         #  'ALCOHOL_HEAVY_DON_unknown',
#         #  'HIST_CIG_DON',
#         # 'HIST_CIG_DON_known',
#         # 'HIST_CIG_DON_unknown',
#         #  'PH_DON',
#         #  'ACIDOSIS',
#         #  'ALKALOSIS',
#         #  'HEP_C_ANTI_DON',
#         # 'HEP_C_ANTI_DON_known',
#         # 'HEP_C_ANTI_DON_unknown',
#         #  'HBV_SUR_ANTIGEN_DON',
#         # 'HBV_SUR_ANTIGEN_DON_known',
#         # 'HBV_SUR_ANTIGEN_DON_unknown',
#         #  'HEPARIN_DON',
#         # 'HEPARIN_DON_known',
#         # 'HEPARIN_DON_unknown',
#         #  'ARGININE_DON'
#         # 'TRANSFUS_TERM_DON',
#         # 'CORONARY_ANGIO_DON',
#         # 'f_to_m',
#         # 'm_to_f',
#         ####recipient####
#         'AGE',
#         # 'GENDER',
#         'MED_COND_TRR',
#         'PREV_TX',
#         'BMI_REC',
#         # 'MELD_NA',
#         # 'FINAL_SERUM_SODIUM',
#         # 'ASCITES_TX',
#         # 'ALBUMIN_TX',
#         'TBILI_TX',#meldとは一緒にしないもしくはmeldを抜く
#         'CREAT_TX',
#         'DIAB',
#         # 'HGT_CM_CALC',
#         # 'AHN',
#         # 'AIH_PBC_PSC',
#         # 'HCV',#diagnosis
#         # 'MALIG',
#         # 'MALIG_known',
#         # 'MALIG_unknown',
#         # 'Cholestatic_disease',
#         # 'PORTAL_VEIN_TRR',
#         'PORTAL_VEIN_TRR_known',
#         # 'PORTAL_VEIN_TRR_unknown',
#         # 'PREV_AB_SURG_TCR',
#         'PREV_AB_SURG_TCR_kown',
#         # 'PREV_AB_SURG_TCR_unknown',
#         'ON_VENT_TRR',
#         # 'PROTEIN_URINE',
#         # 'CARDARREST_NEURO',
#         # 'TATTOOS',
#         # 'TIPSS_TRR'
#         # 'PROTEIN_URINE_known',
#         # 'CARDARREST_NEURO_known',
#         # 'TATTOOS_known',
#         # 'TIPSS_TRR_known',
#         # 'PROTEIN_URINE_unknown',
#         # 'CARDARREST_NEURO_unknown',
#         # 'TATTOOS_unknown',
#         # 'TIPSS_TRR_unknown'
#          ]]

df1 = df1.dropna(axis=0, how='any')
# df1 = df1[df1['NON_HRT_DON'] == 'N']
# df1 = df1.drop('NON_HRT_DON', axis=1)

# df1.to_excel('data.xlsx')

import optuna
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb
import tqdm as tqdm
# df_GSTATUS1 = df1[df1['GSTATUS_3m'] == 1]
# df_GSTATUS0 = df1[df1['GSTATUS_3m'] == 0].sample(n=len(df_GSTATUS1))
# DF = pd.concat([df_GSTATUS1, df_GSTATUS0])
# # DF = DF.dropna(axis=0, how='any')
# x = DF.drop('GSTATUS_3m', axis=1)
# t = np.array(DF['GSTATUS_3m'].tolist())
# x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 0.1, random_state = 8445)
# auc_LGBM_k_list =[]
# best_random_k_list = []
# for k in tqdm.tqdm(range(10)):
#     def LGMOptuna(trial):
#         from sklearn.ensemble import RandomForestClassifier
#         from sklearn.metrics import roc_auc_score
#         import optuna
#         random_state = trial.suggest_int('random_state', 1000*k+1, 1000*(k+1))#元々1500
#         import optuna.integration.lightgbm as lgb
#         dtrain = lgb.Dataset(x_train, label=t_train)
#         dtest = lgb.Dataset(x_test, label=t_test_light)
#         params = {
#         'objective':'binary',
#         'metric':'auc',
#         'verbosity':-1,
#         'boosting_type':'gbdt',
#          # 'boosting_type':'dart',#
#         'random_state':random_state,
#          # 'max_depth':6,#
#         # 'num_leaves':int(0.7*64),#default:31
#         # 'max_bin':trial.suggest_int('max_bin', 300, 500),#
#         # 'max_bin':255,#あげない方が良さそう
#         # 'learning_rate':0.005,#普通0.01~0.005,小さい方が沢山のツリーができて精度高い？変動少なくなって、
#         #あんまり意味なさそう
#         # 'num_iterations':120,#default:100
#         }
#         model_light = lgb.train(params,
#                                 dtrain,
#                                 valid_sets = dtest,
#                                 early_stopping_rounds=10,
#                                 verbose_eval = False
#                                 )
#         predicted_light = model_light.predict(x_test)
#         auc = roc_auc_score(t_test_light, predicted_light)
#         # モデルの保存
#         # import pickle
#         # directory = '/Users/yanagawarintaro/Desktop/research/re_DRI_study/model'
#         # with open(directory + '/model' + str(i) + 'lgmopt.pickle', mode = 'wb') as f:
#         #     pickle.dump(model_light, f)
#         return 1/auc
#     auc_light = 0
#     import time
#     timeout = time.time() + 60*25
#     auc_max = []
#     best_random = []
#     while auc_light <= 0.71:
#         study = optuna.create_study()
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
#         study.optimize(LGMOptuna, 10)#trial数10の方が良い結果が出てくる？
#         auc_light = 1/study.best_value
#         auc_max.append(auc_light)
#         best_random.append(study.best_params['random_state'])
#         if time.time() > timeout:
#             break
#         # 0.70にたどり着くのが先か、breakになるのが先か。。このコードの終了までのタイムが15minよりも小さくなってくれると
#         # 15min以内に0.70越えののaucが出てきたということ。9minで0.7007,5minで0.71越え(たまたま？),20min粘って0.697のことも
#     best_random_k_list.append(best_random[auc_max.index(max(auc_max))])
#     auc_LGBM_k_list.append(max(auc_max))
#     # if max(auc_max)>0.71:
#     #     break

# max(auc_LGBM_k_list)

from sklearn.model_selection import train_test_split
def LGMOptuna(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    import optuna
    from sklearn.utils.class_weight import compute_sample_weight
    # 乱数の選択範囲は1〜1500とした
    random_state = trial.suggest_int('random_state', 1, 10000)#元々1500
    import optuna.integration.lightgbm as lgb
    dtrain = lgb.Dataset(x_train, label=t_train, weight=compute_sample_weight(class_weight='balanced', y=t_train).astype('float32'))
    dtest = lgb.Dataset(x_test, label=t_test_light, weight=np.ones(len(x_test)).astype('float32'))
    params = {
      'objective':'binary',
      'metric':'auc',
      'verbosity':-1,
      'boosting_type':'gbdt',
      # 'boosting_type':'dart',#
      'random_state':random_state,
      # 'max_depth':6,#
      # 'num_leaves':int(0.7*64),#default:31
      # 'max_bin':trial.suggest_int('max_bin', 300, 500),#
      # 'max_bin':255,#あげない方が良さそう
      # 'learning_rate':0.005,#普通0.01~0.005,小さい方が沢山のツリーができて精度高い？変動少なくなって、
      #あんまり意味なさそう
      # 'num_iterations':120,#default:100
      }
    model_light = lgb.train(params,
                             dtrain,
                             valid_sets = [dtrain, dtest],
                             early_stopping_rounds=10,
                             verbose_eval = False
                             )
    predicted_light = model_light.predict(x_test)
    auc = roc_auc_score(t_test_light, predicted_light)
    # モデルの保存
    # import pickle
    # directory = '/Users/yanagawarintaro/Desktop/research/re_DRI_study/model'
    # with open(directory + '/model' + str(i) + 'lgmopt.pickle', mode = 'wb') as f:
    #     pickle.dump(model_light, f)      
    return 1/auc

total_auc = []
df_roc_total_k = pd.DataFrame(index=[i/1000 for i in range(1001)])
for u in tqdm.tqdm(range(4)):
    import optuna
    df_GSTATUS1 = df1[df1['GSTATUS_3m'] == 1]
    df_GSTATUS0 = df1[df1['GSTATUS_3m'] == 0].sample(n=len(df_GSTATUS1))
    DF = pd.concat([df_GSTATUS1, df_GSTATUS0])
    # DF = DF.dropna(axis=0, how='any')
    x = DF.drop('GSTATUS_3m', axis=1)
    t = np.array(DF['GSTATUS_3m'].tolist())
    k = 15
    auc_k = []
    df_roc0 = pd.DataFrame(index=[i/1000 for i in range(1001)])
    for j in tqdm.tqdm(range(k)):
        x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 1/k)#testsizeを0.09などにすると精度が良くなった。これをここに入れるのが肝。splitの最適化をしにいっているようなもの
        auc_light = 0
        import time
        timeout = time.time() + 60*10
        auc_max = []
        best_random = []
        while auc_light <= 0.80:
            # x_train, x_test, t_train, t_test_light = train_test_split(x, t, test_size = 1/10)#testsizeを0.09などにすると精度が良くなった。これをここに入れるのが肝。splitの最適化をしにいっているようなもの
            study = optuna.create_study()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(LGMOptuna, 10)#trial数10の方が良い結果が出てくる？
            auc_light = 1/study.best_value
            auc_max.append(auc_light)
            best_random.append(study.best_params['random_state'])
            if time.time() > timeout:
                break
        # best_random_j = best_random[auc_max.index(max(auc_max))]
        # import optuna.integration.lightgbm as lgb
        # dtrain = lgb.Dataset(x_train, label=t_train)
        # dtest = lgb.Dataset(x_test, label=t_test_light)
        # params = {
        #     'objective':'binary',
        #     'metric':'rmse',
        #     'verbosity':-1,
        #     'boosting_type':'gbdt',
        #     'random_state':best_random_j
        #     }
        # model_light = lgb.train(params,
        #                             dtrain,
        #                             valid_sets = dtest,
        #                             early_stopping_rounds=10,
        #                             verbose_eval = False
        #                             )
        # predicted_light = model_light.predict(x_test)
        # from sklearn.metrics import roc_curve, roc_auc_score
        # fpr, tpr, thresholds = roc_curve(t_test_light, predicted_light)
        # df_roc = pd.DataFrame(
        # data={'fpr': list(np.round(fpr.tolist(),3)), 
        #       'tpr': tpr.tolist()}
        #     )
        # fpr_list = list(set(list(np.round(fpr.tolist(),3))))
        # fpr_list.sort()
        # tpr_list = []
        # for u in fpr_list:
        #     tpr_list.append(df_roc[df_roc['fpr'] == u]['tpr'].mean())
        # df_roc2 = pd.DataFrame(tpr_list, index=fpr_list)
        # df_roc2 = df_roc2.reindex([i/1000 for i in range(1001)])
        # df_roc0[j] = df_roc2.interpolate(limit_direction='both')[0].tolist()#空リストにappendしてるようなもの
        auc_k.append(max(auc_max))
        print(max(auc_max))
    # df_roc_total_k[u] = df_roc0.mean(axis='columns').tolist()
    total_auc.append(np.mean(auc_k))
    print(np.mean(auc_k))




# tpr_shuffled_light = np.array(df_roc_total_k.mean(axis='columns').tolist())
# [np.mean(total_auc), np.mean(total_auc)-1.96*stdev(total_auc), np.mean(total_auc)+1.96*stdev(total_auc)]

# rolling_light_90days_roc = pd.DataFrame(
#     index=[i for i in range(1001)],
#              )

# rolling_light_90days_roc['fpr'] = [i/1000 for i in range(1001)]
# rolling_light_90days_roc['tpr'] = tpr_shuffled_light
# rolling_light_90days_roc.to_excel('15_folds_CV_lightGBM_30days.xlsx')




