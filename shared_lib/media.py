import os
import io
from datetime import datetime
from typing import Union, BinaryIO, Optional
import mimetypes
from .r2 import upload_media


def save_media(filename: str, media: Union[str, BinaryIO, bytes], content_type: Optional[str] = None) -> str:
    """
    Save media (images/videos) to R2 with fallback to local storage.
    
    Args:
        filename (str): The filename for the media
        media (Union[str, BinaryIO, bytes]): The media content - can be a file path, file-like object, or bytes
        content_type (Optional[str]): MIME type of the media. If not provided, will be guessed from filename.
    
    Returns:
        str: URL if saved to R2, or local file path if saved locally
    """
    # Generate date-based directory structure (yy-mm-dd)
    today = datetime.now()
    date_dir = today.strftime("%y-%m-%d")
    
    # Create R2 key with the specified format: seewhy/yy-mm-dd/filename
    r2_key = f"seewhy/{date_dir}/{filename}"
    
    # Try to upload to R2 first
    try:
        # Convert bytes to BytesIO if needed for R2 upload
        if isinstance(media, bytes):
            media_for_upload = io.BytesIO(media)
        else:
            media_for_upload = media
            
        result = upload_media(r2_key, media_for_upload, content_type)
        
        if result['success']:
            print(f"Successfully uploaded to R2: {result['url']}")
            return result['url']
        else:
            print(f"R2 upload failed: {result.get('error', 'Unknown error')}")
            # Fall through to local storage
    except Exception as e:
        print(f"R2 upload error: {str(e)}")
        # Fall through to local storage
    
    # Fallback: Save to local outputs directory
    try:
        # Create outputs directory structure
        local_dir = os.path.join("outputs", date_dir)
        os.makedirs(local_dir, exist_ok=True)
        
        local_path = os.path.join(local_dir, filename)
        
        # Handle different media input types
        if isinstance(media, str):
            # media is a file path
            if os.path.exists(media):
                import shutil
                shutil.copy2(media, local_path)
            else:
                raise FileNotFoundError(f"Source file not found: {media}")
        elif isinstance(media, bytes):
            # media is bytes
            with open(local_path, 'wb') as f:
                f.write(media)
        elif hasattr(media, 'read'):
            # media is a file-like object
            with open(local_path, 'wb') as f:
                f.write(media.read())
        else:
            raise ValueError("Unsupported media type")
        
        print(f"Saved locally: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"Local save failed: {str(e)}")
        raise


def save_matplotlib_figure(filename: str, fig, format: str = 'png', dpi: int = 300) -> str:
    """
    Save a matplotlib figure to R2 with fallback to local storage.
    
    Args:
        filename (str): The filename for the figure (without extension)
        fig: Matplotlib figure object
        format (str): Image format (png, jpg, svg, etc.)
        dpi (int): DPI for raster formats
    
    Returns:
        str: URL if saved to R2, or local file path if saved locally
    """
    # Ensure filename has the correct extension
    if not filename.lower().endswith(f'.{format}'):
        filename = f"{filename}.{format}"
    
    # Save figure to bytes buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    
    # Determine content type
    content_type = mimetypes.guess_type(filename)[0]
    if content_type is None:
        content_type = f'image/{format}'
    
    # Save using the main function
    return save_media(filename, buffer, content_type) 