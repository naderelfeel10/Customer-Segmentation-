import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
import joblib




def apply_ohe(data, col):
    ohe = OneHotEncoder(drop=None, sparse_output=False)
    ohe_enc = ohe.fit_transform(data[[col]])
    ohe_df = pd.DataFrame(ohe_enc, columns=ohe.get_feature_names_out(), index=data.index)
    data_encoded = pd.concat([data.drop(columns=[col]), ohe_df.astype(int)], axis=1)
    return data_encoded, ohe


def apply_ordinal_enc(data, cats, col):
    ord_enc = OrdinalEncoder(categories=cats)
    data[f'{col}_encoding'] = ord_enc.fit_transform(data[[col]])
    return data, ord_enc



def preprocessing_train(data):
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')



    job_saving_map = (data.groupby('Job')['Saving accounts'].agg(lambda x: x.mode()[0])).to_dict()
    data['Saving accounts'] = data['Saving accounts'].fillna(data['Job'].map(job_saving_map))
    job_checking_map = (data.groupby('Job')['Checking account'].agg(lambda x: x.mode()[0])).to_dict()
    data['Checking account'] = data['Checking account'].fillna(data['Job'].map(job_checking_map))



    data, ohe = apply_ohe(data, 'Sex')

    data, ord_housing = apply_ordinal_enc(data, [['free', 'rent', 'own']], 'Housing')

    data, ord_saving = apply_ordinal_enc(data, [['little', 'moderate', 'quite rich', 'rich']], 'Saving accounts')
    data, ord_checking = apply_ordinal_enc(data, [['little', 'moderate', 'rich']], 'Checking account')



    numeric_cols = data.select_dtypes(include='number').copy()
    numeric_cols['Credit amount'] = np.log1p(numeric_cols['Credit amount'])



    scaler = MinMaxScaler()
    numeric_cols[['Age', 'Credit amount', 'Duration']] = scaler.fit_transform(
        numeric_cols[['Age', 'Credit amount', 'Duration']]
    )

    joblib.dump({
        'ohe': ohe,
        'ord_housing': ord_housing,
        'ord_saving': ord_saving,
        'ord_checking': ord_checking,
        'scaler': scaler
    }, 'preprocessors.pkl')



    print(" Preprocessing objects saved to 'preprocessors.pkl'")
    return numeric_cols



def preprocessing_test(data,preprocessors = joblib.load('preprocessors.pkl')):
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')

    
    ohe = preprocessors['ohe']
    ord_housing = preprocessors['ord_housing']
    ord_saving = preprocessors['ord_saving']
    ord_checking = preprocessors['ord_checking']
    scaler = preprocessors['scaler']

    job_saving_map = (data.groupby('Job')['Saving accounts'].agg(lambda x: x.mode()[0])).to_dict()
    data['Saving accounts'] = data['Saving accounts'].fillna(data['Job'].map(job_saving_map))
    job_checking_map = (data.groupby('Job')['Checking account'].agg(lambda x: x.mode()[0])).to_dict()
    data['Checking account'] = data['Checking account'].fillna(data['Job'].map(job_checking_map))

    ohe_enc = ohe.transform(data[['Sex']])
    ohe_df = pd.DataFrame(ohe_enc, columns=ohe.get_feature_names_out(), index=data.index)
    data = pd.concat([data.drop(columns=['Sex']), ohe_df.astype(int)], axis=1)

    data['Housing_encoding'] = ord_housing.transform(data[['Housing']])
    data['saving_encoded'] = ord_saving.transform(data[['Saving accounts']])
    data['checking_encoded'] = ord_checking.transform(data[['Checking account']])

    numeric_cols = data.select_dtypes(include='number').copy()
    numeric_cols['Credit amount'] = np.log1p(numeric_cols['Credit amount'])

    numeric_cols[['Age', 'Credit amount', 'Duration']] = scaler.transform(
        numeric_cols[['Age', 'Credit amount', 'Duration']]
    )

    print("✅ Test data preprocessed using saved encoders/scaler")
    return numeric_cols
