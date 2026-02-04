import requests
import pandas as pd
from sklearn.impute import SimpleImputer

NOAA_TOKEN="yCjjObdzVEXOfvPhRLoVutwtagfOUYdF"
BASE_URL="https://www.ncei.noaa.gov/cdo-web/api/v2/"

headers={"token":NOAA_TOKEN}

dataseturl=BASE_URL+"datasets"
datatypesurl=BASE_URL+"datatypes"
dataurl=BASE_URL+"data"
params={"datasetid":"GHCND",
        "stationid":"GHCND:USW00094728",
        "startdate":"2020-01-01",
        "enddate":"2020-12-31",
        "datatypeid":["PRCP","TMAX","TMIN","TAVG","SNOW","SNWD","AWND","WSF2","WSF5"],
        "limit":1000
        }

response=requests.get(dataurl,headers=headers,params=params)
data=response.json()
df=pd.DataFrame(data["results"])
df=df.pivot(index="date",columns="datatype",values="value").sort_index()
my_imputer=SimpleImputer(strategy="constant",fill_value=0)
imputed_df=pd.DataFrame(my_imputer.fit_transform(df))
imputed_df.columns=df.columns
imputed_df.index=df.index

storm_events_path="StormEvents.csv"

storm_df=pd.read_csv(storm_events_path,parse_dates=["BEGIN_DATE_TIME","END_DATE_TIME"])
storm_df=storm_df[["BEGIN_DATE_TIME","STATE","CZ_NAME","EVENT_TYPE","DAMAGE_PROPERTY","DAMAGE_CROPS"]]

storm_df=storm_df[storm_df["STATE"]=="NEW YORK"]

storm_df["DATE"]=storm_df["BEGIN_DATE_TIME"].dt.date

storm_flags=storm_df.pivot_table(index="DATE",columns="EVENT_TYPE",aggfunc="size",fill_value=0).reset_index()

storm_flags["DATE"]=pd.to_datetime(storm_flags["DATE"],format="%Y-%m-%d")
imputed_df['DATE'] = pd.to_datetime(imputed_df.index)


combined_df=imputed_df.merge(storm_flags,on="DATE",how="left")

missing_cols=storm_flags.columns.drop("DATE")
combined_df[missing_cols]=combined_df[missing_cols].fillna(0)

combined_df=combined_df.set_index("DATE")

if __name__=="__main__":
    print(combined_df.head(100))
