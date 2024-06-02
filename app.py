from typing import List
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from src.file_ingest import read_file
from src.prompt_extractor import extract_pairs, save_to_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import io
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
@app.post("/extract/batch")
async def extract_batch(files: List[UploadFile] = File(...), min_range: float = Query(0.0), max_range: float = Query(1.0)):
    results = []
    for file in files:
        result = process_file(file, min_range, max_range)
        if result:
            results.append(result)
        if results:
            output_path = "batch_output.csv"
            save_to_csv(results, output_path)
            return JSONResponse(content={"message": "Batch extraction successful", "output_file": output_path})
    else:
        return JSONResponse(content={"message": "Failed to read files or empty batch"}, status_code=400)
    

# We are giving an example usage 
file_name = "example.text"
# define a try and exception block
try:
    # some code that might raise FileReadError
    raise FileReadError("FileNotFound", file_name)
except FileReadError as e:
    print(e.message)