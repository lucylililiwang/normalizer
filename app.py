from typing import List
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from src.file_ingest import read_file
from src.prompt_extractor import extract_pairs, save_to_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import io
import chardet
app = FastAPI()

# We are defining the custom exceptions
class FileReadError(Exception):
    # Start with init
    def __init__(self, error_type, file_name):
        self.error_type = error_type
        self.file_name = file_name
        
        # Let's define the first error type: File NotFound error
        if error_type == "FileNotFound":
            self.message = f"Error: File '{file_name}' not found."
        elif error_type == "PermissionError":
            self.message = f"Error: Permission denied while reading '{file_name}'."
        elif error_type == "EmptyFileError":
            self.message = f"Error: File '{file_name}' is empty."
        elif error_type == "InvalidFileFormatError":
            self.message == f"Error: Invalid format or corrupted file '{file_name}'."
        else:
            self.message = f"Error: Unknown error occurred while reading '{file_name}'."
        
        super().__init__(self.message)
    
# we are defining another error handling class for data processing errors
class DataProcessingError(Exception):
    # start with init method
    def __init__(self, error_type, file_name):
        self.error_type = error_type
        self.file_name = file_name
        if error_type == "Data Format Errors":
            self.message = f"Error: File '{file_name}' not found."
        elif error_type == "Missing Values":
            self.message = f"Error: File '{file_name} ' contains missing values." 
        elif error_type == "Outliers":
            self.message = f"Error File '{file_name} have influencial outliers."
        elif error_type == "Data Transformation Errors":
            self.message = f"Error File '{file_name} are having data transformation errors."
        elif error_type == "Algorithmic Errors":
            self.message = f"Error File '{file_name}' are having algorithmic errors."
        elif error_type == "Dimensionality Errors":
            self.message = f"Error File '{file_name}' are having dimensionality errors."
        elif error_type == "Data Integration Errors":
            self.message = f"Error File '{file_name}' are having data integration errors."
        elif error_type == "Concurrency and Parallelism Errors":
            self.message = f"Error File '{file_name}' are having concurrency and parallelism error."
        elif error_type == "Resource Limit Errors":
            self.message = f"Error File '{file_name}' are having resource limit errors. "
        elif error_type == "Domain-specific Errors":
            self.message = f"Error File '{file_name}' are having domain-specific errors."
        else:
            self.message = f"Error: Unknown error occurred while reading '{file_name}'."
            
        
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    text = read_file(content)
    if text:
        pairs = extract_pairs(text)
        # We wish to convert pairs to a DataFrame
        df = pd.DataFrame(pairs, columns=["feature1","feature2"])
        # We are Apply Z-score normalizer
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df)
        # We are Convert normalized data back to DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
        output_path = "output.csv"
        save_to_csv(normalized_df.tolist(), output_path)
        return JSONResponse(content={"message": "Extraction successful", "output_file": output_path})
    else:
        return JSONResponse(content={"message": "Failed to read file"}, status_code=400)
# We also need to define a class for general error handling
class GeneralError(Exception):
    def __init__(self, error_type, file_name):
        self.error_type = error_type
        self.file_name = file_name
        
        if error_type:
            self.message = f"General Error: {error_type} occurred while processing '{file_name}'."
        else:
            self.message = f"General Error occurred while processing '{file_name}'."
        
        super().__init__(self.message)
        
        
@app.post("/extract/minmax")
async def extract_minmax(file:UploadFile = File(...),min_range: float = Query(0.0),max_range: float = Query(1.0)):
    content = await file.read()
    text = read_file(content)
    if text:
        pairs = extract_pairs(text)
        # We are convert pairs to DataFrame
        df = pd.DataFrame(pairs, columns = ["featue1","feature2"])
        # We are apply min-max scaling
        scaler = MinMaxScaler(feature_range=(min_range,max_range))
        scaled_data = scaler.fit_transform(df)
        # We are Convert scaled data back to DataFrame
        scaled_df = pd.DataFrame(scaled_data,columns=df.columns)
        
        output_path = "minmax_output.csv"
        save_to_csv(scaled_df.values.tolist(), output_path)
        return JSONResponse(content={"message": "Min-Max scaling successful", "output_file": output_path})
    else:
        return JSONResponse(content={"message": "Failed to read file"}, status_code=400)
# We also need to create a function to check whether is missing deimiler
def is_missing_delimiter(file_content, delimier=','):
    """
    Check if a CSV file is missing delimiters
    

    Args:
        - file_content (str): Content of the CSV file as a string.
        - delimiter (str): Delimiter used in the CSV file (default is comma ',').

    Returns:
        - bool: True is missing delimiter are detected, False otherwise.
        
    """
    # We are split the file content into lines
    lines = file_content.splitlines()
    # We are check the number of fields in each line
    for line in lines:
        fields = line.split(delimier)
        if len(fields) < 2:
            # The missing delimiter detected
            return True
    return False # No missing delimiter detected
    
# define another function to check for encoding
def detect_encoding(file_content):
    """
    Detect the encoding of file content

    Args:
        - file_content (bytes): Content of the file as bytes.

    
    Returns:
       - str: Detected encoding of the file content.
    """
    # We are detect the encoding of the file content
    encoding_info = chardet.detect(file_content)
    detected_encoding = encoding_info['encoding']
    confidence = encoding_info['confidence']

    # We are checking if the detected encoding has high confidence, return it
    if detect_encoding and confidence > 0.5:
        return detect_encoding
    else:
        # We are checking if the confidence is low or no encoding is detected, return None
        return None
    
# We need to define another function to recover data
def recover_data_missing_delimiter(file_content):
    """
    Recover data from a CSV file with missing delimiters.

    Args:
        - file_content (str): Content of the CSV file as a string

    Returns:
       list of lists: Recovered data where each sublist represents a row of the CSV.
    """
    # We are check if the file content is missing delimiters
    if is_missing_delimiter(file_content):
        # We are attempt to recover daya by inferring the delimiter
        inferred_delimiter = inferred_delimiter(file_content)
        # We are split lines using the inferred delimiter
        lines = file_content.splitlines()
        recovered_data = [line.split(inferred_delimiter) for line in lines]
        return recovered_data
    else:
        #If delimiters are present, split lines using comma as the default delimiter
        lines = file_content.splitlines()
        recovered_data = [line.split(',') for line in lines]
        return recovered_data
    
# after defining the error class, we need to have some method to fix the error
# We need to define a function to fix error
def fix_error_delimiter(error_type, file_name):
    # we need to implement logic to fix or mitigate errord
    if error_type == "InvalidFormatError":
        # we need to attempt to recover data from the invalid format
        recovered_data = recover_data_missing_delimiter(file_name)
        # We are returing the recovered data
        return recovered_data
    
# We need create another function to take care of the encoding problem
def recover_date_encoding(file_content):
    """
    Recover data encoding from  byte string.

    Args:
        file_content (bytes): Content of the file as bytes.

    
    Returns:
        str or None: Detected encoding of the file content, or NOne if encoding detection failed.
    """
    # We are define a try block
    try:
        # We are detect the encoding of the file content
        encoding_info = chardet.detect(file_content)
        detected_encoding = encoding_info['encoding']
        confidence = encoding_info['confidence']
        
        if detect_encoding and confidence > 0.5:
            return detected_encoding
        else:
            return None
    except Exception as e:
        # We are handle encoding detection errors
        print(f"Error detecting encoding: {e}")
        return None
    
# We need to create another function to check for incomplete lines
def has_incomplete_lines(file_content, delimiter=','):
    """
    Check if a CSV file has incoplete lines or records.

    Args:
        file_content (str): Content of the CSV file as a string.
        delimiter (str): Delimiter used in the CSV file, Default is ','.

    
    Returns:
        bool: True if the last line of the file ends with a delimiter, indicating an incomplete line, False otherwise.
    """
    # We are Split the file content nto lines
    lines = file_content.splitines()
    # We are get the ast line
    last_line = lines[-1] if lines else ''
    # We are check if the last line ends with the delimiter
    return last_line.endswith(delimiter)

# We are creating a function to recover incomplete lines
def recover_incomplete_lines(file_content, delimiter=','):
    """
    Recover data from a CSV file with incomplete lines or records

    Args:
        file_content (str): Content of the CSV file as a string.
        delimiter (str):  Delimiter used in the CSV file, Default is ','.
    

    Returns:
       list of lists: Recovered data where each sublist represents a row of  the CSV.
       
    """
    recovered_data = []
    incomplete_line = ''
    
    lines = file_content.splitlines()
    for line in lines:
        # We are Concatenate the current line with any leftoer incomplete line from the previous iterations
        line = incomplete_line + line
        # We are split the line using the specified delimiter
        line_data = line.split(delimiter)
        # We are check if the line is complete
        if line.endswith(delimiter):
            # We are checking if the line ends with the delimiter, int's incomplete
            # We are stroe the incomplete line for the next iteration
            incomplete_line = line
        else:
            #if the line is complete, append it to the recovered data
            recovered_data.append(line_data)
            incomplete_line = ''
    # We are check if there's any leftover incomplete line
    if incomplete_line:
        # We are check if there's an incomplate line remaining after processing all lines,
        # it's considered a separate incomplete record
        recovered_data.append(incomplete_line.split(delimiter))
    return recovered_data

@app.post("/extract/categorical_variable_handling")   
async def extract(file: UploadFile = File(...),min_range: float = Query(0.0), max_range: float = Query(1.0)):
    content = await file.read()
    text = read_file(content)
    if text:
        pairs = extract_pairs(text)
        # We are convert pairs to a DataFrame
        df = pd.DataFrame(pairs, columns= ["categorical_feature", "numeric_feature"])
        # We are separate categorical and numeric features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numeric_features = df.select_dtypes(include=['number'])
        # We are apply one-hot encoding to categorical features
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
        # We are Concatenate encoded categorical features with numeric features
        df_encoded = pd.concat([encoded_df,df[numeric_features]], axis=1)
        # We are apply min-max scaling with customizable range
        scaler = MinMaxScaler(feature_range=(min_range, max_range))
        scaled_data = scaler.fit_transform(df_encoded)
        # We are convert scaled data back to DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=df_encoded.columns)
        output_path = "output.csv"
        save_to_csv(scaled_df.values.tolist(), output_path)
        return JSONResponse(content={"message": "Extraction successful", "output_file": output_path})
    else:
        return JSONResponse(content={"message": "Failed to read file"}, status_code=400)
        


# We wish to add another post for batch jobs
# @app.post("/extract/batch")
# async def extract_batch(files: List[UploadFile] = File(...), min_range: float = Query(0.0), max_range: float = Query(1.0)):
#     results = []
#     for file in files:
#         result = process_file(file, min_range, max_range)
#         if result:
#             results.append(result)
#         if results:
#             output_path = "batch_output.csv"
#             save_to_csv(results, output_path)
#             return JSONResponse(content={"message": "Batch extraction successful", "output_file": output_path})
#     else:
#         return JSONResponse(content={"message": "Failed to read files or empty batch"}, status_code=400)
    

# We are giving an example usage 
file_name = "example.text"
# define a try and exception block
try:
    # some code that might raise FileReadError
    raise FileReadError("FileNotFound", file_name)
except FileReadError as e:
    print(e.message)
    
