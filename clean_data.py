import pandas as pd

# Load scraped data
df = pd.read_csv("shl_assessments.csv")

# Remove duplicates
df = df.drop_duplicates()

# Drop rows with missing names
df = df.dropna(subset=["assessment_name"])

# Standardize text
df["assessment_name"] = df["assessment_name"].str.strip()

# Save cleaned file
df.to_csv("shl_assessments_clean.csv", index=False)

print("Cleaning completed successfully!")
print("Total assessments after cleaning:", len(df))