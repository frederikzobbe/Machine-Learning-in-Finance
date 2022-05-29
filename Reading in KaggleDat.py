import pandas as pd

# Tilføj dit eget directory til den synkroniserede google drive mappe, så kan filer indlæses hurtigt
MichaelDirectory = str("Final project/data")

# Ændrer til dit eget directory
UsedDirectory = MichaelDirectory

# Indlæsning af Kaggle data
TestFinancials = pd.read_csv(UsedDirectory + "/KaggleDat/jpx-tokyo-stock-exchange-prediction/example_test_files/financials.csv")
