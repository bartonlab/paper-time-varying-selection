#!/bin/bash
cd ..
python inference_HIV.py -tag 703010256-3 -dir 'data/HIV250' --raw || echo "CH256-3 failed, continuing with next script..." 
python inference_HIV.py -tag 703010256-5 -dir 'data/HIV250' --raw || echo "CH256-5 failed, continuing with next script..." 
python inference_HIV.py -tag 704010042-3 -dir 'data/HIV250' --raw || echo "CH042-3 failed, continuing with next script..." 
python inference_HIV.py -tag 704010042-5 -dir 'data/HIV250' --raw || echo "CH042-5 failed, continuing with next script..." 
python inference_HIV.py -tag 705010162-3 -dir 'data/HIV250' --raw || echo "CH162-3 failed, continuing with next script..." 
python inference_HIV.py -tag 705010162-5 -dir 'data/HIV250' --raw || echo "CH162-5 failed, continuing with next script..." 
python inference_HIV.py -tag 706010164-3 -dir 'data/HIV250' --raw || echo "CH164-3 failed, continuing with next script..." 
python inference_HIV.py -tag 706010164-5 -dir 'data/HIV250' --raw || echo "CH164-5 failed, continuing with next script..." 
echo "All scripts have been attempted."