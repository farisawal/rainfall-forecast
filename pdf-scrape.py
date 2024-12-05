# Install doctr and other dependencies if not already installed
# !pip install python-doctr[torch] pandas

import pandas as pd
import re
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load the OCR model
ocr_model = ocr_predictor(pretrained=True)
folder_name = "mulu"
# Specify your PDF file path
pdf_path = "Department of Irrigation and Drainage (DID)/Rainfall data from DID Sarawak/4048001.pdf"  # Replace with your PDF file path
# re.search(r"rainfall-data\d+\.pdf", file)
pdf_doc = DocumentFile.from_pdf(pdf_path)

# Initialize the output CSV file and create the header
csv_output_path = "mulu_rainfall_data.csv"
with open(csv_output_path, 'w') as f:
    # Date	Time	Rainfall (mm)-Long Merarap	Quality Code	Station ID
    f.write("Station ID,Value,DateTime,Rainfall (mm),Quality Code\n")  # Create headers for the CSV

# Process each page one by one
for page_idx, page in enumerate(pdf_doc, start=1):
    # Perform OCR on the current page
    result = ocr_model([page])
    # print(result)
    # Extract text from the page
    for block in result.pages[0].blocks:
        for line in block.lines:
            text_line = ' '.join([word.value for word in line.words])
            # Assume data rows are properly comma-separated and contain 5 values
            data_row = text_line.split(",")  # Split the row by commas
            
            if len(data_row) == 5:  # If the row contains exactly 5 elements
                # Create a DataFrame for the row and append it to the CSV
                df = pd.DataFrame([data_row], columns=["Station ID","Value","DateTime","Rainfall (mm)","Quality Code"])
                df.to_csv(csv_output_path, mode='a', header=False, index=False)
    
    print(f"Processed Page {page_idx} and appended to CSV")

print(f"PDF data successfully saved to {csv_output_path}")
