"""
Configuration module for managing environment variables and secrets.
"""

import os
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables."""
    
    # MT5 Configuration
    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None
    MT5_PATH: Optional[str] = None
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        try:
            # MT5 Login (convert to int if provided)
            login_str = os.getenv('MT5_LOGIN')
            if login_str:
                self.MT5_LOGIN = int(login_str)
            
            # MT5 Password
            self.MT5_PASSWORD = os.getenv('MT5_PASSWORD')
            
            # MT5 Server
            self.MT5_SERVER = os.getenv('MT5_SERVER')
            
            # MT5 Terminal Path
            self.MT5_PATH = os.getenv('MT5_PATH')
            
            # Log Level
            self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
            
            logger.info("Configuration loaded successfully")
            
        except ValueError as e:
            logger.error(f"Error parsing configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def validate_mt5_config(self) -> bool:
        """
        Validate that required MT5 configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if self.MT5_PATH:
            # If path is provided, login credentials are optional
            return True
        
        # If no path, require login credentials
        if not all([self.MT5_LOGIN, self.MT5_PASSWORD, self.MT5_SERVER]):
            logger.error("Missing required MT5 configuration. Please check your .env file.")
            return False
        
        return True
    
    def get_mt5_credentials(self) -> dict:
        """
        Get MT5 credentials as a dictionary.
        
        Returns:
            dict: MT5 credentials
        """
        credentials = {}
        
        if self.MT5_LOGIN:
            credentials['login'] = self.MT5_LOGIN
        if self.MT5_PASSWORD:
            credentials['password'] = self.MT5_PASSWORD
        if self.MT5_SERVER:
            credentials['server'] = self.MT5_SERVER
        if self.MT5_PATH:
            credentials['path'] = self.MT5_PATH
            
        return credentials

# Global configuration instance
config = Config()
