import os
import requests
import hashlib
import hmac
from datetime import datetime
from typing import Optional, Union, BinaryIO
import mimetypes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# R2 Configuration from environment variables
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')
R2_BASE_URL = os.getenv('R2_BASE_URL')

def _validate_config():
    """Validate that all required environment variables are set."""
    required_vars = ['R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_ENDPOINT_URL', 'R2_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def _get_signature_key(key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    """Generate AWS signature key for authentication."""
    k_date = hmac.new(f"AWS4{key}".encode('utf-8'), date_stamp.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region_name.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service_name.encode('utf-8'), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
    return k_signing

def _sign(string_to_sign: str, signing_key: bytes) -> str:
    """Sign a string using the provided signing key."""
    return hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

def _get_aws_signature_v4(method: str, uri: str, headers: dict, payload: bytes = b'') -> dict:
    """Generate AWS Signature V4 for R2 authentication."""
    # Validate configuration
    _validate_config()
    
    # AWS Signature V4 components
    algorithm = 'AWS4-HMAC-SHA256'
    region = 'auto'  # R2 uses 'auto' as the region
    service = 's3'
    
    # Get current timestamp
    t = datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')
    
    # Canonical request
    canonical_uri = uri
    canonical_querystring = ''
    
    # Sort headers
    canonical_headers = ''
    signed_headers = ''
    for key in sorted(headers.keys()):
        canonical_headers += f"{key.lower()}:{headers[key]}\n"
        signed_headers += f"{key.lower()};"
    signed_headers = signed_headers.rstrip(';')
    
    # Payload hash
    payload_hash = hashlib.sha256(payload).hexdigest()
    
    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    
    # String to sign
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    
    # Calculate signature
    signing_key = _get_signature_key(R2_SECRET_ACCESS_KEY, date_stamp, region, service)
    signature = _sign(string_to_sign, signing_key)
    
    # Authorization header
    authorization_header = f"{algorithm} Credential={R2_ACCESS_KEY_ID}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    
    return {
        'Authorization': authorization_header,
        'X-Amz-Date': amz_date,
        'X-Amz-Content-Sha256': payload_hash
    }

def upload_media(r2_key: str, file_path: Union[str, BinaryIO], content_type: Optional[str] = None) -> dict:
    """
    Upload a file to Cloudflare R2 storage.
    
    Args:
        r2_key (str): The key/path where the file will be stored in R2
        file_path (Union[str, BinaryIO]): Path to the file or file-like object
        content_type (Optional[str]): MIME type of the file. If not provided, will be guessed from file extension.
    
    Returns:
        dict: Response containing upload status and URL
    """
    try:
        # Validate configuration
        _validate_config()
        
        # Handle file input
        if isinstance(file_path, str):
            # File path provided
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Guess content type if not provided
            if content_type is None:
                content_type, _ = mimetypes.guess_type(file_path)
                if content_type is None:
                    content_type = 'application/octet-stream'
        else:
            # File-like object provided
            file_data = file_path.read()
            if hasattr(file_path, 'name') and content_type is None:
                content_type, _ = mimetypes.guess_type(file_path.name)
                if content_type is None:
                    content_type = 'application/octet-stream'
            elif content_type is None:
                content_type = 'application/octet-stream'
        
        # Prepare headers
        headers = {
            'Content-Type': content_type,
            'Content-Length': str(len(file_data)),
            'Host': R2_ENDPOINT_URL.replace('https://', ''),
        }
        
        # Generate AWS Signature V4
        signature_headers = _get_aws_signature_v4('PUT', f'/{R2_BUCKET_NAME}/{r2_key}', headers, file_data)
        headers.update(signature_headers)
        
        # Make the request
        url = f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}/{r2_key}"
        response = requests.put(url, data=file_data, headers=headers)
        
        if response.status_code == 200:
            return {
                'success': True,
                'url': f"{R2_BASE_URL}/{r2_key}",
                'key': r2_key,
                'content_type': content_type,
                'size': len(file_data),
                'status_code': response.status_code
            }
        else:
            return {
                'success': False,
                'error': f"Upload failed with status code {response.status_code}",
                'response_text': response.text,
                'status_code': response.status_code
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': None
        }

def list_objects(prefix: str = '', max_keys: int = 1000) -> dict:
    """
    List objects in the R2 bucket.
    
    Args:
        prefix (str): Prefix to filter objects
        max_keys (int): Maximum number of keys to return
    
    Returns:
        dict: Response containing list of objects
    """
    try:
        # Validate configuration
        _validate_config()
        
        # Prepare query parameters
        params = {
            'list-type': '2',
            'max-keys': str(max_keys)
        }
        if prefix:
            params['prefix'] = prefix
        
        canonical_querystring = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Prepare headers
        headers = {
            'Host': R2_ENDPOINT_URL.replace('https://', ''),
        }
        
        # Generate AWS Signature V4
        signature_headers = _get_aws_signature_v4('GET', f'/{R2_BUCKET_NAME}', headers)
        headers.update(signature_headers)
        
        # Make the request
        url = f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}?{canonical_querystring}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return {
                'success': True,
                'objects': response.text,  # XML response
                'status_code': response.status_code
            }
        else:
            return {
                'success': False,
                'error': f"List failed with status code {response.status_code}",
                'response_text': response.text,
                'status_code': response.status_code
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': None
        }

def delete_object(r2_key: str) -> dict:
    """
    Delete an object from R2 storage.
    
    Args:
        r2_key (str): The key of the object to delete
    
    Returns:
        dict: Response containing delete status
    """
    try:
        # Validate configuration
        _validate_config()
        
        # Prepare headers
        headers = {
            'Host': R2_ENDPOINT_URL.replace('https://', ''),
        }
        
        # Generate AWS Signature V4
        signature_headers = _get_aws_signature_v4('DELETE', f'/{R2_BUCKET_NAME}/{r2_key}', headers)
        headers.update(signature_headers)
        
        # Make the request
        url = f"{R2_ENDPOINT_URL}/{R2_BUCKET_NAME}/{r2_key}"
        response = requests.delete(url, headers=headers)
        
        if response.status_code == 204:
            return {
                'success': True,
                'key': r2_key,
                'status_code': response.status_code
            }
        else:
            return {
                'success': False,
                'error': f"Delete failed with status code {response.status_code}",
                'response_text': response.text,
                'status_code': response.status_code
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'status_code': None
        } 