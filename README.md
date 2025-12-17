# Vast.ai GPU CLI

CLI tool for managing GPU instances on Vast.ai marketplace.

## Setup

```bash
# 1. Clone and enter directory
cd gpu

# 2. Install dependencies
make install

# 3. Configure API key
cp .env.example .env
# Edit .env and add your API key from https://cloud.vast.ai/account/
```

## Usage

### Search GPUs

```bash
# List all available GPUs (cheapest first)
make search

# Filter by max price
make search PRICE=0.15

# Filter by GPU model
make search GPU=RTX_4090

# Combine filters
make search GPU=RTX_4090 PRICE=0.20

# Quick search: cheap GPUs with best price/performance
make cheap              # Under $0.04/hr, sorted by value
make cheap PRICE=0.06   # Override price limit
```

### Launch Instances

```bash
# Launch by offer ID (from search results)
make launch ID=12345

# Launch multiple instances
make launch ID=12345,67890

# Use different Docker image
make launch ID=12345 IMAGE=tensorflow
```

Available images: `pytorch` (default), `tensorflow`, `cuda`, `cuda_devel`

### Manage Instances

```bash
# List your instances
make list

# Get SSH command
make ssh ID=12345

# Start/stop instance
make start ID=12345
make stop ID=12345

# Destroy instance
make destroy ID=12345

# Destroy all instances
make destroy-all        # with confirmation
make destroy-all-force  # skip confirmation
```

### Account Info

```bash
# Check balance
make balance

# View billing history
make billing

# List available Docker images
make images
```

## All Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make search` | Search available GPUs |
| `make cheap` | Search cheap GPUs (<$0.04, best value) |
| `make list` | List your instances |
| `make launch ID=X` | Launch instance by offer ID |
| `make ssh ID=X` | Get SSH command |
| `make start ID=X` | Start stopped instance |
| `make stop ID=X` | Stop running instance |
| `make destroy ID=X` | Destroy instance |
| `make destroy-all` | Destroy all instances |
| `make balance` | Show account balance |
| `make billing` | Show billing history |
| `make images` | List Docker images |

## Development

```bash
# Install dev tools (pre-commit, ruff)
make dev

# Run linter
make lint

# Format code
make format

# Run all checks
make check
```
