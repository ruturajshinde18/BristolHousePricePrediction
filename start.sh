#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port 9002 &
streamlit run app/app.py --server.port 9502 --server.address 0.0.0.0
