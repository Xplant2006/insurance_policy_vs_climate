from mastertable import master_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBRegressor,XGBClassifier
import pandas as pd
import numpy as np


features=["PRCP_7D_SUM","TMAX_30D_MEAN","WSF5","AWND","TMAX","TMIN","PRCP","CoverageAmount","PropertyType","SNOW","SNWD","WSF2"]

X=master_df[features]
y_class=master_df["ClaimOccurred"]
y_reg=master_df.loc[master_df["ClaimOccurred"]==1,"ClaimAmount"]
x_reg=X.loc[master_df["ClaimOccurred"]==1]

num_features = ['AWND','PRCP','SNOW','SNWD','TMAX','TMIN','WSF2','WSF5','PRCP_7D_SUM','TMAX_30D_MEAN','CoverageAmount']
cat_features = ['PropertyType']

preprocessor=ColumnTransformer([("num",StandardScaler(),num_features),("cat",OneHotEncoder(handle_unknown="ignore"),cat_features)])

X_train,X_test,y_train,y_test=train_test_split(X,y_class,test_size=0.2,random_state=0,stratify=y_class)

clf_pipeline=Pipeline([("preprocess",preprocessor),("model",XGBClassifier(n_estimators=500,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss"))])

clf_pipeline.fit(X_train,y_train)

master_df["ClaimProbability"]=clf_pipeline.predict_proba(X)[:,1]

y_reg_log=np.log1p(y_reg)

reg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

reg_pipeline.fit(x_reg,y_reg_log)

pred_log=reg_pipeline.predict(X)
master_df["PredictedSeverity"]=np.expm1(pred_log)

policy_risk = master_df.groupby("PolicyID").apply(
    lambda x: (x["ClaimProbability"] * x["PredictedSeverity"]).sum() / x["CoverageAmount"].iloc[0]
)

policy_risk = policy_risk.reset_index().rename(columns={0: "OverallRiskScore"})

scaler=MinMaxScaler()
policy_risk["NormalizedRisk"]=scaler.fit_transform(policy_risk[["OverallRiskScore"]])

kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
policy_risk["Cluster"]=kmeans.fit_predict(policy_risk[["NormalizedRisk"]])

cluster_order=(policy_risk.groupby("Cluster")["NormalizedRisk"].mean().sort_values().index)
cluster_mapping={old: i for i, old in enumerate(cluster_order)}
policy_risk["RiskLevel"]=policy_risk["Cluster"].map(cluster_mapping)

risk_labels={0:"Low Risk",1:"Medium Risk",2:"High Risk"}
policy_risk["RiskCategory"]=policy_risk["RiskLevel"].map(risk_labels)
policy_risk["RiskScore100"]=(policy_risk["NormalizedRisk"]*100).round(2)

policy_data = master_df.drop_duplicates(subset="PolicyID")
policy_data = policy_data.merge(policy_risk[["PolicyID","RiskCategory"]],on="PolicyID")

X1=policy_data[features]
y1=policy_data["RiskCategory"]
encoder=LabelEncoder()
y1_encoded=encoder.fit_transform(y1)

num_features = ['AWND','PRCP','SNOW','SNWD','TMAX','TMIN','WSF2','WSF5','PRCP_7D_SUM','TMAX_30D_MEAN','CoverageAmount']
cat_features = ['PropertyType']

preprocessor1=ColumnTransformer([("num",StandardScaler(),num_features),("cat",OneHotEncoder(handle_unknown="ignore"),cat_features)])

X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1_encoded,test_size=0.2,random_state=42,stratify=y1_encoded)

clf_final=Pipeline([("preprocess",preprocessor1),("model",XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,eval_metric="mlogloss",random_state=42))])
clf_final.fit(X1_train,y1_train)

y_pred=clf_final.predict(X1_test)
y_pred_labels=encoder.inverse_transform(y_pred)

if __name__=="__main__":
    print(confusion_matrix(y1_test,y_pred))
    print(classification_report(y1_test,y_pred))



