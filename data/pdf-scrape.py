# Install doctr and other dependencies if not already installed
# !pip install python-doctr[torch] pandas

import pandas as pd
import re
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load the OCR model
ocr_model = ocr_predictor(pretrained=True)
folder_name = "mulu"
pdf_path = "mulu/4048001-2.pdf"

# Initialize OCR
pdf_doc = DocumentFile.from_pdf(pdf_path)

# Initialize the output CSV file
csv_output_path = f"{folder_name}/rainfall_data2.csv"
with open(csv_output_path, 'w') as f:
    f.write("Station ID,Value,DateTime,Rainfall (mm),Quality Code\n")

def process_data_rows(rows):
    """Process rows handling both single-line and split-line formats."""
    combined_rows = []
    i = 0
    while i < len(rows):
        current_row = rows[i]
        
        # Case 1: Complete row with all 5 elements
        if len(current_row) == 5:
            combined_rows.append([
                current_row[0].strip(),  # Station ID
                current_row[1].strip(),  # Value
                current_row[2].strip(),  # DateTime
                current_row[3].strip(),  # Rainfall
                current_row[4].strip()   # Quality Code
            ])
            i += 1
            
        # Case 2: Split format (station info + rainfall/quality code)
        elif len(current_row) >= 3 and current_row[0].strip().startswith('4048001'):
            if i + 1 < len(rows) and len(rows[i+1]) == 2:
                rainfall_row = rows[i+1]
                combined_rows.append([
                    current_row[0].strip(),    # Station ID
                    current_row[1].strip(),    # Value
                    current_row[2].strip(),    # DateTime
                    rainfall_row[0].strip(),   # Rainfall
                    rainfall_row[1].strip()    # Quality Code
                ])
                i += 2
            else:
                # Skip invalid or incomplete data
                i += 1
        else:
            # Skip any other unrecognized format
            i += 1
            
    return combined_rows

# Process each page
for page_idx, page in enumerate(pdf_doc, start=1):
    # Store rows for the current page
    current_page_rows = []
    
    # Perform OCR on the current page
    result = ocr_model([page])
    
    # Extract text from the page
    for block in result.pages[0].blocks:
        for line in block.lines:
            text_line = ' '.join([word.value for word in line.words])
            # Remove any empty strings that might appear after splitting
            data_row = [item.strip() for item in text_line.split(',') if item.strip()]
            if data_row:  # Only add non-empty rows
                current_page_rows.append(data_row)
    
    # Process the collected rows
    combined_rows = process_data_rows(current_page_rows)
    print(combined_rows)
    
    # Write to CSV
    if combined_rows:  # Only write if we have data
        df = pd.DataFrame(combined_rows, columns=["Station ID", "Value", "DateTime", "Rainfall (mm)", "Quality Code"])
        df.to_csv(csv_output_path, mode='a' if page_idx > 1 else 'w', header=(page_idx == 1), index=False)
        print(f"Processed Page {page_idx} and appended {len(combined_rows)} rows to CSV")
    else:
        print(f"No valid data found on Page {page_idx}")

print(f"PDF data successfully saved to {csv_output_path}")