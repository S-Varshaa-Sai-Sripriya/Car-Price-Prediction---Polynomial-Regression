# test_ingestion.py
from src.Data_Wrangling.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    path = ingestion.download_data()
    csv_file = ingestion.get_csv_path()
    print("CSV File path:", csv_file)
