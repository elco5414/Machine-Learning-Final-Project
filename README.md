# Machine-Learning-Final-Project
### Portfolio Predictor

**Group Members:**
- Mariana Vadas-Arendt
- Elizabeth Coleman

**Hugging Face Hosting for Application**

[Hosted Application Link](https://huggingface.co/spaces/elco5414/portfolio-prediction)

## How to Use
**Dependencies**
- Python 3.10, 3.11, 3.12, or 3.13. **Python 3.14 is not yet supported**: `pandas-ta` pins `numba==0.61.2`, which does not support 3.14 as of this writing.
- On macOS, OpenMP must be installed for XGBoost: `brew install libomp`.
- for all specific package dependencies they are in `requirements.txt` and you can install them via `pip install -r requirements.txt`

**general installation**s
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*If your shell has an alias or path quirk that causes `python` or `pip` to resolve outside the venv (like me), use the venv executables explicitly, you have to use abs path*

```bash
/opt/homebrew/bin/python3.12 -m venv venv     
source venv/bin/activate
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```
If you had to use absolute path for installation - you have to continue to use that path through out also. 

Also if you are not familiar with virtual environments, you use `deactivate` to exit from it. 

**training**
- of course, if you are using the exact repo, you do not need to do these again, bc they have already been trained, so if you want to go nuts but its not a requirenment

```bash
# 1. download and process data -> writes data/training_data.parquet
python data_pipeline.py

# 2. train the LSTM -> writes models/model.pth and models/scaler.npy
python model.py

# 3. train the XGBoost model -> writes models/price_model.json
python price_model.py
```

**running web-app locally**
```bash
python -m uvicorn api:app --port 8000
```
Then open `http://localhost:8000` in a browser

If all these things are not calling to you - it is hosted on hugging face and you can see/use it there too (see link above).
****

**note**
- Do not use this for your actual investment advice, this is for academics
- AI was used for the ideation of these models and the generation of model code

If you want to use `visualize_tree.py`, which shows you the actual tree graphs at 0 and the final one, you need to also 'brew install graphviz`, and then it will populate those pngs. But they will be in the repo so you can just look there. 
****
