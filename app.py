from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.file_ingest import read_file
from src.prompt_extractor import extract_pairs, save_to_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import io
app = FastAPI()

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
    
@app.post("/extract/minmax")
async def extract_minmax(file:UploadFile = File(...)):
    content = await file.read()
    text = read_file(content)
    if text:
        pairs = extract_pairs(text)
        # We are convert pairs to DataFrame
        df = pd.DataFrame(pairs, columns = ["featue1","feature2"])
        # We are apply min-max scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        # We are Convert scaled data back to DataFrame
        scaled_df = pd.DataFrame(scaled_data,columns=df.columns)
        
        output_path = "minmax_output.csv"
        save_to_csv(scaled_df.values.tolist(), output_path)
        return JSONResponse(content={"message": "Min-Max scaling successful", "output_file": output_path})
    else:
        return JSONResponse(content={"message": "Failed to read file"}, status_code=400)