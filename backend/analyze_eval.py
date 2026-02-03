import pandas as pd

file_path = r"C:\Users\NakulSaiAdapala\Downloads\enterprise_search_v0.3\docs\LIC_QA_Evaluation_Results.xlsx"

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("-" * 50)
    for index, row in df.head(5).iterrows():
        print(f"Row {index}:")
        for col in df.columns:
            print(f"  {col}: {str(row[col])[:200]}...") # truncate long text
        print("-" * 20)

except Exception as e:
    print(f"Error reading excel: {e}")
