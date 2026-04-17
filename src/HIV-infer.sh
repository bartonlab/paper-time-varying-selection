#!/bin/bash
cd ..
python infer_HIV.py -tag 700010058-3 || echo "CH058-3 failed, continuing with next script..." 
python infer_HIV.py -tag 705010185-5 || echo "CH185-5 failed, continuing with next script..." 
python infer_HIV.py -tag 700010077-3 || echo "CH077-3 failed, continuing with next script..." 
python infer_HIV.py -tag 700010607-3 || echo "CH607-3 failed, continuing with next script..." 
python infer_HIV.py -tag 705010198-3 || echo "CH198-3 failed, continuing with next script..." 
python infer_HIV.py -tag 703010159-3 || echo "CH159-3 failed, continuing with next script..." 
echo "All scripts have been attempted."