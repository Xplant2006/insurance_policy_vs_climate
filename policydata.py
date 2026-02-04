import pandas as pd
from faker import Faker
import numpy as np
from climatedata import combined_df

SEED=42

Faker.seed(SEED)

fake=Faker()
Faker.seed(SEED)
np.random.seed(SEED)


num_policies=1000
policies=pd.DataFrame({
    "PolicyID":[fake.uuid4() for _ in range(num_policies)],
    "HolderName":[fake.name() for _ in range(num_policies)],
    "State":"New York",
    "PropertyType":np.random.choice(["residential","commercial","farm"],num_policies),
    "CoverageAmount":np.random.randint(100_000,1_000_000,num_policies),
    "StartDate":pd.to_datetime(np.random.choice(combined_df.index,num_policies))
})

policies["EndDate"]=policies["StartDate"]+pd.to_timedelta(np.random.randint(365,730,num_policies),unit="D")

claims_list=[]

for _, row in policies.iterrows():
    policy_id=row["PolicyID"]
    policy_dates=pd.date_range(row["StartDate"],row["EndDate"])
    num_claims=np.random.poisson(lam=0.2)

    if num_claims>0:
        possible_dates=np.random.choice(policy_dates,num_claims)

        for claim_date in possible_dates:
            if claim_date in combined_df.index:
                day=combined_df.loc[claim_date]
                event_probs={
                    "flood":0.1 if day["PRCP"]>20 else 0.01,
                    "hail":0.1 if day.get("Hail",0)>0 else 0.01,
                    "wind":0.1 if day.get("High Wind",0)>0 else 0.01
                }
            else:
                event_probs={"flood":0.01,"hail":0.01,"wind":0.01}
            
            claim_types=list(event_probs.keys())
            probs=list(event_probs.values())
            claim_type=np.random.choice(claim_types,p=np.array(probs)/sum(probs))

            claim_amount=int(row["CoverageAmount"]*np.random.uniform(0.05,0.5))

            claims_list.append({
                "Name":row["HolderName"],
                "ClaimID":fake.uuid4(),
                "PolicyID":policy_id,
                "ClaimDate":claim_date,
                "ClaimType":claim_type,
                "ClaimAmount":claim_amount,
            })

claims=pd.DataFrame(claims_list)

if __name__=="__main__":
    print(claims)
