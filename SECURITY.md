# Secret Management Guide

This document explains how to securely manage passwords and other sensitive information in your MT5 Python Bot project.

## 🔐 Security Setup

### 1. Install Dependencies

First, install the required package for environment variable management:

```bash
pip install python-dotenv
```

### 2. Create Environment File

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit the `.env` file with your actual MT5 credentials:
   ```env
   # MT5 Trading Account Configuration
   MT5_LOGIN=12345678
   MT5_PASSWORD=your_actual_password
   MT5_SERVER=YourBroker-Demo
   
   # Optional: MT5 Terminal Path
   # MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   
   # Other Configuration
   LOG_LEVEL=INFO
   ```

### 3. Usage in Code

The configuration is automatically loaded when you import the config module:

```python
from data_handler import initialize_mt5_with_config
from config import config

# Initialize MT5 with credentials from .env file
if initialize_mt5_with_config():
    print("Connected successfully!")
```

## 🛡️ Security Best Practices

### ✅ DO:
- Use the `.env` file for local development
- Use environment variables in production
- Keep different `.env` files for different environments (dev, staging, prod)
- Use strong, unique passwords
- Regularly rotate your credentials

### ❌ DON'T:
- Commit `.env` files to version control
- Share credentials via email or chat
- Use the same password for multiple accounts
- Store credentials in code comments

## 🌍 Production Deployment

For production environments, set environment variables directly:

### Windows (PowerShell):
```powershell
$env:MT5_LOGIN="12345678"
$env:MT5_PASSWORD="your_password"
$env:MT5_SERVER="YourBroker-Live"
```

### Linux/Mac:
```bash
export MT5_LOGIN="12345678"
export MT5_PASSWORD="your_password"
export MT5_SERVER="YourBroker-Live"
```

### Docker:
```dockerfile
ENV MT5_LOGIN=12345678
ENV MT5_PASSWORD=your_password
ENV MT5_SERVER=YourBroker-Live
```

## 🔄 Alternative Approaches

### 1. Azure Key Vault (for Azure deployments)
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)

mt5_password = client.get_secret("mt5-password").value
```

### 2. AWS Systems Manager (for AWS deployments)
```python
import boto3

ssm = boto3.client('ssm')
response = ssm.get_parameter(Name='/mt5/password', WithDecryption=True)
mt5_password = response['Parameter']['Value']
```

### 3. Configuration Files (less secure)
```python
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
mt5_password = config['MT5']['password']
```

## 🚨 Emergency Procedures

If credentials are compromised:

1. **Immediately change your MT5 password**
2. **Update the `.env` file with new credentials**
3. **Check your trading account for unauthorized activity**
4. **Review access logs if available**
5. **Consider using 2FA if your broker supports it**

## 📝 Configuration Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MT5_LOGIN` | Yes* | MT5 account number | `12345678` |
| `MT5_PASSWORD` | Yes* | MT5 account password | `SecurePass123!` |
| `MT5_SERVER` | Yes* | Broker server name | `YourBroker-Demo` |
| `MT5_PATH` | No | MT5 terminal path | `C:\Program Files\MetaTrader 5\terminal64.exe` |
| `LOG_LEVEL` | No | Logging level | `INFO`, `DEBUG`, `WARNING`, `ERROR` |

*Required unless `MT5_PATH` is specified for local terminal access.
