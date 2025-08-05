"""
Advanced Secrets Management System for Cyber-LLM
Provides secure handling of API keys, credentials, and sensitive configuration
"""

import os
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import hvac  # HashiCorp Vault client
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from google.cloud import secretmanager
import yaml
import logging

from .logging_system import CyberLLMLogger, CyberLLMError, ErrorCategory

class SecretNotFoundError(CyberLLMError):
    """Raised when a secret is not found"""
    def __init__(self, secret_name: str):
        super().__init__(
            f"Secret not found: {secret_name}",
            ErrorCategory.AUTHENTICATION,
            details={"secret_name": secret_name}
        )

class SecretDecryptionError(CyberLLMError):
    """Raised when secret decryption fails"""
    def __init__(self, secret_name: str):
        super().__init__(
            f"Failed to decrypt secret: {secret_name}",
            ErrorCategory.SECURITY,
            details={"secret_name": secret_name}
        )

class SecretProvider:
    """Base class for secret providers"""
    
    def __init__(self, logger: Optional[CyberLLMLogger] = None):
        self.logger = logger or CyberLLMLogger(name="secrets")
    
    async def get_secret(self, name: str) -> str:
        """Get a secret by name"""
        raise NotImplementedError
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set a secret value"""
        raise NotImplementedError
    
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret"""
        raise NotImplementedError
    
    async def list_secrets(self) -> List[str]:
        """List available secrets"""
        raise NotImplementedError

class LocalEncryptedProvider(SecretProvider):
    """Local encrypted file-based secret provider"""
    
    def __init__(self, 
                 secrets_file: str = "secrets/encrypted_secrets.json",
                 password: Optional[str] = None,
                 logger: Optional[CyberLLMLogger] = None):
        
        super().__init__(logger)
        self.secrets_file = Path(secrets_file)
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate or load encryption key
        self.password = password or os.getenv("CYBERLLM_SECRETS_PASSWORD")
        if not self.password:
            raise ValueError("Secrets password must be provided via parameter or CYBERLLM_SECRETS_PASSWORD env var")
        
        self.cipher_suite = self._get_cipher_suite()
        
        # Load existing secrets
        self.secrets = self._load_secrets()
    
    def _get_cipher_suite(self) -> Fernet:
        """Generate cipher suite from password"""
        password_bytes = self.password.encode()
        salt = b'cyberllm_secrets_salt'  # In production, use random salt per secret
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from encrypted file"""
        if not self.secrets_file.exists():
            return {}
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return {}
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        
        except Exception as e:
            self.logger.error("Failed to load secrets", error=str(e))
            return {}
    
    def _save_secrets(self) -> bool:
        """Save secrets to encrypted file"""
        try:
            data = json.dumps(self.secrets).encode()
            encrypted_data = self.cipher_suite.encrypt(data)
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            return True
        
        except Exception as e:
            self.logger.error("Failed to save secrets", error=str(e))
            return False
    
    async def get_secret(self, name: str) -> str:
        """Get a secret by name"""
        if name not in self.secrets:
            raise SecretNotFoundError(name)
        
        self.logger.audit("Secret accessed", secret_name=name)
        return self.secrets[name]
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set a secret value"""
        self.secrets[name] = value
        success = self._save_secrets()
        
        if success:
            self.logger.audit("Secret updated", secret_name=name)
        else:
            self.logger.error("Failed to save secret", secret_name=name)
        
        return success
    
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret"""
        if name in self.secrets:
            del self.secrets[name]
            success = self._save_secrets()
            
            if success:
                self.logger.audit("Secret deleted", secret_name=name)
            
            return success
        
        return False
    
    async def list_secrets(self) -> List[str]:
        """List available secrets"""
        return list(self.secrets.keys())

class VaultProvider(SecretProvider):
    """HashiCorp Vault secret provider"""
    
    def __init__(self,
                 vault_url: str = "http://localhost:8200",
                 vault_token: Optional[str] = None,
                 mount_point: str = "kv",
                 logger: Optional[CyberLLMLogger] = None):
        
        super().__init__(logger)
        self.vault_url = vault_url
        self.mount_point = mount_point
        
        # Initialize Vault client
        self.client = hvac.Client(url=vault_url)
        
        # Authenticate
        token = vault_token or os.getenv("VAULT_TOKEN")
        if token:
            self.client.token = token
        else:
            # Try other auth methods (AWS IAM, Kubernetes, etc.)
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Vault using available methods"""
        # Try AWS IAM authentication
        if os.getenv("AWS_ROLE"):
            try:
                self.client.auth.aws.iam_login(
                    role=os.getenv("AWS_ROLE"),
                    mount_point="aws"
                )
                self.logger.info("Authenticated with Vault using AWS IAM")
                return
            except Exception as e:
                self.logger.warning("AWS IAM auth failed", error=str(e))
        
        # Try Kubernetes authentication
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            try:
                with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
                    jwt = f.read()
                
                self.client.auth.kubernetes.login(
                    role=os.getenv("VAULT_K8S_ROLE", "cyberllm"),
                    jwt=jwt,
                    mount_point="kubernetes"
                )
                self.logger.info("Authenticated with Vault using Kubernetes")
                return
            except Exception as e:
                self.logger.warning("Kubernetes auth failed", error=str(e))
        
        raise CyberLLMError("Failed to authenticate with Vault", ErrorCategory.AUTHENTICATION)
    
    async def get_secret(self, name: str) -> str:
        """Get a secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=name,
                mount_point=self.mount_point
            )
            
            secret_data = response['data']['data']
            
            # If secret has a 'value' key, return that, otherwise return the first value
            if 'value' in secret_data:
                value = secret_data['value']
            else:
                value = next(iter(secret_data.values()))
            
            self.logger.audit("Secret accessed from Vault", secret_name=name)
            return value
        
        except hvac.exceptions.InvalidPath:
            raise SecretNotFoundError(name)
        except Exception as e:
            self.logger.error("Failed to get secret from Vault", 
                            secret_name=name, error=str(e))
            raise CyberLLMError(f"Vault error: {str(e)}", ErrorCategory.SYSTEM)
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set a secret in Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=name,
                secret={'value': value},
                mount_point=self.mount_point
            )
            
            self.logger.audit("Secret updated in Vault", secret_name=name)
            return True
        
        except Exception as e:
            self.logger.error("Failed to set secret in Vault", 
                            secret_name=name, error=str(e))
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete a secret from Vault"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=name,
                mount_point=self.mount_point
            )
            
            self.logger.audit("Secret deleted from Vault", secret_name=name)
            return True
        
        except Exception as e:
            self.logger.error("Failed to delete secret from Vault", 
                            secret_name=name, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List secrets in Vault"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="",
                mount_point=self.mount_point
            )
            return response['data']['keys']
        
        except Exception as e:
            self.logger.error("Failed to list secrets from Vault", error=str(e))
            return []

class AWSSecretsProvider(SecretProvider):
    """AWS Secrets Manager provider"""
    
    def __init__(self,
                 region_name: str = "us-east-1",
                 logger: Optional[CyberLLMLogger] = None):
        
        super().__init__(logger)
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    async def get_secret(self, name: str) -> str:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=name)
            
            self.logger.audit("Secret accessed from AWS", secret_name=name)
            return response['SecretString']
        
        except self.client.exceptions.ResourceNotFoundException:
            raise SecretNotFoundError(name)
        except Exception as e:
            self.logger.error("Failed to get secret from AWS", 
                            secret_name=name, error=str(e))
            raise CyberLLMError(f"AWS error: {str(e)}", ErrorCategory.SYSTEM)
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in AWS Secrets Manager"""
        try:
            try:
                # Try to update existing secret
                self.client.update_secret(SecretId=name, SecretString=value)
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                self.client.create_secret(Name=name, SecretString=value)
            
            self.logger.audit("Secret updated in AWS", secret_name=name)
            return True
        
        except Exception as e:
            self.logger.error("Failed to set secret in AWS", 
                            secret_name=name, error=str(e))
            return False
    
    async def delete_secret(self, name: str) -> bool:
        """Delete secret from AWS Secrets Manager"""
        try:
            self.client.delete_secret(
                SecretId=name,
                ForceDeleteWithoutRecovery=True
            )
            
            self.logger.audit("Secret deleted from AWS", secret_name=name)
            return True
        
        except Exception as e:
            self.logger.error("Failed to delete secret from AWS", 
                            secret_name=name, error=str(e))
            return False
    
    async def list_secrets(self) -> List[str]:
        """List secrets in AWS Secrets Manager"""
        try:
            paginator = self.client.get_paginator('list_secrets')
            secrets = []
            
            for page in paginator.paginate():
                for secret in page['SecretList']:
                    secrets.append(secret['Name'])
            
            return secrets
        
        except Exception as e:
            self.logger.error("Failed to list secrets from AWS", error=str(e))
            return []

class SecretsManager:
    """Central secrets manager with fallback providers"""
    
    def __init__(self, 
                 providers: Optional[List[SecretProvider]] = None,
                 cache_ttl: int = 300,  # 5 minutes
                 logger: Optional[CyberLLMLogger] = None):
        
        self.logger = logger or CyberLLMLogger(name="secrets_manager")
        self.providers = providers or []
        self.cache_ttl = cache_ttl
        self.cache = {}
        
        # Auto-configure providers based on environment
        if not self.providers:
            self._auto_configure_providers()
    
    def _auto_configure_providers(self):
        """Auto-configure providers based on available credentials/environment"""
        
        # Try Vault if available
        if os.getenv("VAULT_ADDR") or os.getenv("VAULT_TOKEN"):
            try:
                vault_provider = VaultProvider(
                    vault_url=os.getenv("VAULT_ADDR", "http://localhost:8200"),
                    logger=self.logger
                )
                self.providers.append(vault_provider)
                self.logger.info("Added Vault provider")
            except Exception as e:
                self.logger.warning("Failed to configure Vault provider", error=str(e))
        
        # Try AWS if credentials available
        try:
            boto3.Session().get_credentials()
            aws_provider = AWSSecretsProvider(logger=self.logger)
            self.providers.append(aws_provider)
            self.logger.info("Added AWS Secrets Manager provider")
        except Exception as e:
            self.logger.warning("Failed to configure AWS provider", error=str(e))
        
        # Always add local encrypted provider as fallback
        if os.getenv("CYBERLLM_SECRETS_PASSWORD"):
            try:
                local_provider = LocalEncryptedProvider(logger=self.logger)
                self.providers.append(local_provider)
                self.logger.info("Added local encrypted provider")
            except Exception as e:
                self.logger.warning("Failed to configure local provider", error=str(e))
    
    def _is_cache_valid(self, name: str) -> bool:
        """Check if cached secret is still valid"""
        if name not in self.cache:
            return False
        
        cache_entry = self.cache[name]
        return datetime.now() < cache_entry['expires']
    
    async def get_secret(self, name: str, use_cache: bool = True) -> str:
        """Get secret from providers with caching"""
        
        # Check cache first
        if use_cache and self._is_cache_valid(name):
            self.logger.debug("Secret retrieved from cache", secret_name=name)
            return self.cache[name]['value']
        
        # Try each provider
        for i, provider in enumerate(self.providers):
            try:
                value = await provider.get_secret(name)
                
                # Cache the result
                if use_cache:
                    self.cache[name] = {
                        'value': value,
                        'expires': datetime.now() + timedelta(seconds=self.cache_ttl),
                        'provider': type(provider).__name__
                    }
                
                self.logger.info(f"Secret retrieved from provider {i+1}/{len(self.providers)}", 
                               secret_name=name, 
                               provider=type(provider).__name__)
                return value
            
            except SecretNotFoundError:
                continue  # Try next provider
            except Exception as e:
                self.logger.warning(f"Provider {type(provider).__name__} failed", 
                                  secret_name=name, error=str(e))
                continue
        
        # No provider found the secret
        raise SecretNotFoundError(name)
    
    async def set_secret(self, name: str, value: str) -> bool:
        """Set secret in the first available provider"""
        
        # Clear cache
        if name in self.cache:
            del self.cache[name]
        
        for provider in self.providers:
            try:
                success = await provider.set_secret(name, value)
                if success:
                    self.logger.info("Secret set successfully", 
                                   secret_name=name,
                                   provider=type(provider).__name__)
                    return True
            except Exception as e:
                self.logger.warning(f"Provider {type(provider).__name__} failed to set secret", 
                                  secret_name=name, error=str(e))
                continue
        
        return False
    
    def get_api_credentials(self) -> Dict[str, str]:
        """Get common API credentials"""
        import asyncio
        
        credentials = {}
        
        common_secrets = [
            "metasploit_api_key",
            "cobalt_strike_license",
            "shodan_api_key",
            "virustotal_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "github_token",
            "slack_webhook_url",
            "discord_webhook_url"
        ]
        
        for secret in common_secrets:
            try:
                value = asyncio.run(self.get_secret(secret))
                credentials[secret] = value
            except SecretNotFoundError:
                self.logger.debug(f"Optional secret not found: {secret}")
            except Exception as e:
                self.logger.warning(f"Failed to get {secret}", error=str(e))
        
        return credentials

# Global instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    
    return _secrets_manager

# Convenience functions
async def get_secret(name: str) -> str:
    """Get a secret using the global manager"""
    return await get_secrets_manager().get_secret(name)

async def set_secret(name: str, value: str) -> bool:
    """Set a secret using the global manager"""
    return await get_secrets_manager().set_secret(name, value)

def get_api_credentials() -> Dict[str, str]:
    """Get API credentials using the global manager"""
    return get_secrets_manager().get_api_credentials()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_secrets():
        # Initialize secrets manager
        manager = SecretsManager()
        
        # Set a test secret
        await manager.set_secret("test_api_key", "super_secret_value_123")
        
        # Retrieve the secret
        value = await manager.get_secret("test_api_key")
        print(f"Retrieved secret: {value}")
        
        # Get API credentials
        credentials = manager.get_api_credentials()
        print(f"Available credentials: {list(credentials.keys())}")
    
    asyncio.run(test_secrets())
