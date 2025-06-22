#!/bin/bash


conda activate llama2
cd llserver/server
uvicorn uniserver:app --reload --host 0.0.0.0 --port 8000