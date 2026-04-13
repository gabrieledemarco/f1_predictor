"""
Rate Limiter Utility

Provides rate limiting and retry logic for API calls.
Handles 429 Too Many Requests errors gracefully.
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimiter:
    """Rate limiter with exponential backoff for API calls."""
    
    def __init__(
        self,
        requests_per_second: float = 4.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        retry_on_status: tuple = (429, 500, 502, 503, 504),
    ):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_on_status = retry_on_status
        self._last_request_time = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to maintain rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()
    
    def create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=self.retry_on_status,
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Make a request with rate limiting and retry logic."""
        self.wait()
        
        session = self.create_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = session.request(method, url, **kwargs)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    log.warning(
                        f"Rate limited (429). Waiting {retry_after}s before retry "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(retry_after)
                    continue
                
                if response.status_code >= 500:
                    wait_time = self.backoff_factor ** attempt
                    log.warning(
                        f"Server error {response.status_code}. Waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    time.sleep(wait_time)
                    continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                wait_time = self.backoff_factor ** attempt
                log.warning(
                    f"Request failed: {e}. Waiting {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(wait_time)
        
        raise requests.exceptions.RetryError(
            f"Max retries ({self.max_retries}) exceeded for {url}"
        )


def with_rate_limit(requests_per_second: float = 4.0):
    """Decorator to add rate limiting to a function."""
    limiter = RateLimiter(requests_per_second=requests_per_second)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator


DEFAULT_LIMITER = RateLimiter(requests_per_second=4.0)
