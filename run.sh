#!/bin/bash

PORT=8080

# 檢查埠號是否被佔用
PID=$(lsof -ti:$PORT)

if [ -n "$PID" ]; then
  echo "Port $PORT is in use by process $PID. Killing it..."
  kill -9 $PID
  echo "Process $PID has been terminated."
else
  echo "Port $PORT is not in use."
fi

# 啟動 app.py
echo "Starting app.py..."
python app.py