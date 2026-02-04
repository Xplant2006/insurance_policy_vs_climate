import pandas as pd
from policydata import claims,policies
from climatedata import combined_df
from datetime import timedelta


climate_df=combined_df.copy()
climate_df.reset_index(inplace=True)
climate_df.rename(columns={'index':'DATE'},inplace=True)
climate_df["DATE"]=pd.to_datetime(climate_df["DATE"])

records=[]

for _,row in policies.iterrows():
    policy_days=pd.date_range(row["StartDate"]-timedelta(days=90),row["EndDate"])
    for d in policy_days:
        records.append({
            "DATE":d,
            "PolicyID":row["PolicyID"],
            "State":"New York",
            "PropertyType":row["PropertyType"],
            "CoverageAmount":row["CoverageAmount"],
            "StartDate":row["StartDate"],
            "EndDate":row["EndDate"]
        })

policy_daily=pd.DataFrame(records)

policy_daily=policy_daily.merge(climate_df,on="DATE",how="left")

claims["ClaimDate"]=pd.to_datetime(claims["ClaimDate"])

claims_daily=claims.groupby(["PolicyID","ClaimDate"]).agg(ClaimAmount=("ClaimAmount","sum")).reset_index().rename(columns={"ClaimDate":"DATE"})

claims_daily["ClaimOccurred"]=1

master_df=policy_daily.merge(
    claims_daily,
    on=["PolicyID","DATE"],
    how="left"
)

master_df["ClaimOccurred"]=master_df["ClaimOccurred"].fillna(0).astype(int)
master_df["ClaimOccurred"]=master_df["ClaimOccurred"].fillna(0)
master_df["ClaimAmount"]=master_df["ClaimAmount"].fillna(0)

master_df=master_df.sort_values(["PolicyID","DATE"])

master_df["PRCP_7D_SUM"]=master_df.groupby("PolicyID")["PRCP"].transform(lambda x:x.rolling(7,min_periods=1).sum())
master_df["TMAX_30D_MEAN"]=master_df.groupby("PolicyID")["TMAX"].transform(lambda x:x.rolling(30,min_periods=1).mean())
master_df=master_df[master_df["DATE"]<=master_df["EndDate"]]

if __name__ =="__main__":
    print(master_df.head(10))

    print(master_df["ClaimOccurred"].value_counts())

    master_df.to_csv("master_df.csv")