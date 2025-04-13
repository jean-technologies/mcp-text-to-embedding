# Text-to-Embedding MCP Server

This project provides a service to convert text to vector embeddings using OpenAI's embedding models and perform similarity searches. It's designed to be used as a standalone CLI, or as a Model Context Protocol (MCP) server for integration with Claude Desktop and other MCP clients.

## Features

- Generate embeddings from text using OpenAI's embedding models
- Save embeddings with custom IDs
- Perform similarity searches across stored embeddings
- Compare semantic similarity between two texts
- Delete and manage stored embeddings
- MCP server integration for Claude Desktop

## Requirements

- Python 3.9+ (for standalone CLI)
- Python 3.10+ (for MCP server)
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mcp-text-to-embedding.git
   cd mcp-text-to-embedding
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key.

## API Key Security

⚠️ **IMPORTANT**: Your OpenAI API key should be kept secure. Never commit it to version control.

- The `.gitignore` file is set up to exclude the `.env` file from Git
- Use `.env.example` as a template, but create your own `.env` file locally
- For production deployments, consider using a secrets management service
- If you accidentally commit your API key, you should immediately rotate it on the OpenAI dashboard

## Usage

### Standalone CLI

The `embedding_cli.py` script provides a command-line interface for working with embeddings:

```
# Generate an embedding and save it
./embedding_cli.py generate "This is a test sentence" --id test-sentence

# List all saved embeddings
./embedding_cli.py list

# Compare two texts for similarity
./embedding_cli.py compare "This is a test" "This is a sample"

# Search for similar embeddings
./embedding_cli.py search "test query" --top-k 3 --threshold 0.7

# Delete an embedding
./embedding_cli.py delete test-sentence
```

Run `./embedding_cli.py --help` for more information on each command.

### MCP Server (Python 3.10+ required)

To run as an MCP server (requires Python 3.10+ and the MCP package):

1. Ensure you have Python 3.10+ installed
2. Install the MCP package: `pip install "mcp[cli]"`
3. Run the MCP server: `python mcp_server.py`

## Claude Desktop Integration

To use this server with Claude Desktop:

1. Make sure Claude Desktop is installed (download from [anthropic.com](https://www.anthropic.com/claude))
2. Create a Claude Desktop configuration file at `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Configure the server using the example in `claude_desktop_config.example.json`:

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

Make sure to:
- Replace `/absolute/path/to/mcp-text-to-embedding` with the actual path to your project
- Replace `your_openai_api_key_here` with your actual OpenAI API key
- Restart Claude Desktop after making changes

## Available MCP Tools

When run as an MCP server, the following tools are available:

- `text_to_embedding`: Convert text to an embedding vector and save it
- `similarity_search`: Find similar embeddings to a query text
- `compare_texts`: Compare two text strings and get a similarity score with interpretation
- `list_embeddings`: Show all saved embeddings
- `delete_embedding`: Remove an embedding from the repository

## Project Structure

- `text_to_embedding.py`: Core embedding generation functionality
- `similarity_search.py`: Functions for similarity search and embedding management
- `embedding_cli.py`: Command-line interface for working with embeddings
- `mcp_server.py`: MCP server implementation
- `embeddings/`: Directory where embeddings are stored

## About Embeddings

Text embeddings are vector representations of text that capture semantic meaning. They're useful for:

- Semantic search
- Text clustering
- Finding similar documents
- Building recommendation systems
- And more!

## About MCP

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between LLM applications and external data sources and tools. Learn more at [modelcontextprotocol.io](https://modelcontextprotocol.io).

## License

MIT
