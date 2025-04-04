from fastapi import Header, status
from fastapi.responses import JSONResponse

from .models import Accept, ContentType


MOCK_API_KEY = "mock-api-key"


def check_auth(
    access_token: str = Header(None, alias="X-Access-Token"),
    api_key: str = Header(None, alias="X-Api-Key"),
) -> JSONResponse | None:
    if access_token is None and api_key is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "400",
                    "title": "Bad Request",
                    "message": "Neither X-Access-Token nor X",
                }
            },
        )

    if access_token is not None and api_key is not None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "400",
                    "title": "Bad Request",
                    "message": "Choose either X-Access-Token or X-Api-Key",
                }
            },
        )

    # NOTE: This mock server only authorizes `"X-Api-Key": MOCK_API_KEY`
    if api_key is None or api_key != MOCK_API_KEY:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "401",
                    "title": "Unauthorized",
                    "message": "Access denied due to invalid header",
                }
            },
        )

    return None


def check_accept(
    accept: Accept = Header(None, alias="Accept"),
) -> JSONResponse | None:
    """
    NOTE:
    This check does not exist in DA4, but was added because we want to make accept: application/json mandatory.
    """
    if accept is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "400",
                    "title": "Bad Request",
                    "message": "Accept header is required",
                }
            },
        )

    if accept != Accept.json:
        return JSONResponse(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            content={
                "error": {
                    "code": "406",
                    "title": "Not Acceptable",
                    "message": "Accept header is not application/json",
                }
            },
        )

    return None


def check_content_type(
    content_type: ContentType = Header(None, alias="Content-Type"),
) -> JSONResponse | None:
    """
    NOTE:
    This check does not exist in DA4, but was added because we want to make Content-Type: application/json mandatory.
    """
    if content_type is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "400",
                    "title": "Bad Request",
                    "message": "Content-Type header is required",
                }
            },
        )

    if content_type != ContentType.json:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "error": {
                    "code": "415",
                    "title": "Unsupported Media Type",
                    "message": "Content-Type header is not application/json",
                }
            },
        )

    return None
