cd ../data
mkdir features model_data save_model submit log
mkdir features/lgb_emb
cd ../code
python3 fea_process.py
python3 get_data.py
python3 lgb.py
python3 nn.py
python3 lgb_inference.py
python3 nn_inference.py
python3 inference.py

