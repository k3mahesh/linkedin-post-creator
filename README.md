# linkedin-post-creator


```
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new directory for our project
uv init weather
cd weather

# Create virtual environment and activate it
uv venv
source .venv/bin/activate
```


MCP Server Configuration
- Download Clint like claude

```
{
  "mcpServers": {
    "linked-post-creator": {
      "command": "/Users/opstree/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/opstree/mine/github/linkedin-post-creator",
        "run",
        "mcp_server.py"
      ]
    }
  }
}
```