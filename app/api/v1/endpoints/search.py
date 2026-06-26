from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.search import SearchResponse
from search import search_main

router = APIRouter()


@router.get("/", response_model=SearchResponse, summary="Search treatments and devices")
def run_search(
    query: str = Query(..., min_length=1, description="The search query text"),
    debug: bool = Query(False, description="Whether to include extra logs for search diagnostics"),
    limit: int = Query(5, ge=1, le=50, description="Max results per category")
):
    """
    Execute semantic and classification-based search against treatments and devices.
    """
    try:
        results = search_main(query=query, debug=debug, limit=limit)
        return results
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search pipeline error: {str(e)}"
        )
