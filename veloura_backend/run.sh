#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/.."

echo "=================================================="
echo "    Veloura Agentic AI - Backend & Dashboard     "
echo "=================================================="

# Check virtual environment
if [ ! -d "veloura_backend/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv veloura_backend/venv
fi

echo "Activating virtual environment..."
source veloura_backend/venv/bin/activate

echo "Checking dependencies..."
pip install -q -r veloura_backend/requirements.txt

echo ""
echo "🚀 Starting Veloura Backend Server..."
echo "📊 Clinical Dashboard:  http://localhost:8000/dashboard/"
echo "📖 Swagger API Docs:    http://localhost:8000/docs"
echo "📈 Productivity Stats:  http://localhost:8000/api/productivity/metrics"
echo ""

uvicorn veloura_backend.main:app --host 0.0.0.0 --port 8000 --reload
