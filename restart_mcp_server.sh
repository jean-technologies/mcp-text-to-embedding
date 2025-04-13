#!/bin/bash
# Kill any existing MCP server processes
pkill -f "python mcp_server.py" || true
echo "Stopped existing MCP server processes"

# Navigate to project directory
cd /Users/jonathanpolitzki/Desktop/Coding/mcp-text-to-embedding

# Activate virtual environment and start server
source venv/bin/activate
echo "Starting MCP server..."
python mcp_server.py 