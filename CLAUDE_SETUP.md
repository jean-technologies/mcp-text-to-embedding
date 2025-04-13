# Setting Up with Claude Desktop

This guide explains how to set up the Text-to-Embedding MCP server with Claude Desktop.

## Requirements

- Claude Desktop app (download from [anthropic.com](https://www.anthropic.com/claude))
- Python 3.10 or higher
- MCP package (`pip install "mcp[cli]"`)

## Setup Steps

1. **Install Python 3.10+**:
   
   This MCP server requires Python 3.10 or higher. You can check your version with:
   ```
   python --version
   ```
   
   If you're using a version below 3.10, you'll need to install a newer version of Python.

2. **Install the MCP package**:
   
   ```
   pip install "mcp[cli]"
   ```

3. **Configure Claude Desktop**:
   
   Create or edit the Claude Desktop configuration file at:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```
   
   Add the following configuration (using your actual paths and API key):
   ```json
   {
       "mcpServers": {
           "text-embeddings": {
               "command": "python",
               "args": [
                   "/absolute/path/to/mcp-text-to-embedding/mcp_server.py"
               ],
               "env": {
                   "OPENAI_API_KEY": "your_openai_api_key_here"
               },
               "cwd": "/absolute/path/to/mcp-text-to-embedding"
           }
       }
   }
   ```

4. **Restart Claude Desktop**:
   
   After making these changes, completely quit and restart Claude Desktop.

5. **Test the integration**:
   
   In Claude Desktop, you should now see a hammer icon that allows you to access the tools. Try using the tools with prompts like:
   
   - "Generate an embedding for 'This is a test sentence' and save it as 'test-1'"
   - "Compare the similarity between 'I love machine learning' and 'I enjoy artificial intelligence'"
   - "Find embeddings similar to 'machine learning is interesting'"
   - "List all my saved embeddings"

## Troubleshooting

If the MCP server doesn't appear in Claude Desktop:

1. **Check your Python version**:
   Make sure you're using Python 3.10+.

2. **Verify the configuration file**:
   Check that your `claude_desktop_config.json` file has the correct format and paths.

3. **Check for error messages**:
   Look for error messages in the Claude Desktop logs.

4. **Test the server directly**:
   Try running the server manually to see if there are any errors:
   ```
   python mcp_server.py
   ```

5. **Check permissions**:
   Make sure Claude Desktop has permission to execute the Python script.

## Additional Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Claude Desktop Documentation](https://www.anthropic.com/claude) 