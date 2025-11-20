"""
Authentication utilities for API key validation
"""
from fastapi import Depends, HTTPException, Header, status
from typing import Optional
from config import get_settings

settings = get_settings()


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=settings.API_KEY_HEADER)
) -> str:
    """
    Verify API key from request header

    Args:
        x_api_key: API key from request header

    Returns:
        The valid API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Skip authentication if disabled
    if not settings.ENABLE_API_KEY_AUTH:
        return "anonymous"

    # Check if API key is provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if API key is valid
    if x_api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key
