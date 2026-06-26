import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from app.schemas.tags import SingleTagRequest, SingleTagResponse, BatchTagRequest, BatchTagResponse
from batch_tag_generator import batch_tag_generator_main
from tag_entity import tag_entity_main

router = APIRouter()


@router.post("/batch", response_model=BatchTagResponse, status_code=status.HTTP_202_ACCEPTED, summary="Batch generate tags")
def batch_generate_tags(
    request: BatchTagRequest,
    background_tasks: BackgroundTasks
):
    """
    Run batch tag generation across all or specific treatments/devices.
    Can be run in the background (recommended) to avoid request timeouts.
    """
    if request.background:
        task_id = str(uuid.uuid4())
        
        # Define a wrapper to capture exceptions in the background log
        def run_in_background():
            try:
                batch_tag_generator_main(
                    dry_run=request.dry_run,
                    entity_type=request.entity_type,
                    limit=request.limit,
                    force=request.force
                )
            except Exception as e:
                # Logs will be written to stderr/stdout
                print(f"Background batch tagging task {task_id} failed: {e}")

        background_tasks.add_task(run_in_background)
        return BatchTagResponse(
            message="Batch tag generation successfully started in background.",
            task_id=task_id
        )
    else:
        try:
            results = batch_tag_generator_main(
                dry_run=request.dry_run,
                entity_type=request.entity_type,
                limit=request.limit,
                force=request.force
            )
            return BatchTagResponse(
                message="Batch tag generation completed synchronously.",
                total_success=results.get("total_success"),
                total_skipped=results.get("total_skipped"),
                total_failed=results.get("total_failed")
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Configuration error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch tag generation pipeline error: {str(e)}"
            )


@router.post("/entity/{entity_id}", response_model=SingleTagResponse, summary="Tag single treatment/device")
def tag_single_entity(
    entity_id: str,
    request: SingleTagRequest
):
    """
    Generate or regenerate tags for a single treatment or device using its UUID.
    """
    try:
        result = tag_entity_main(
            entity_id=entity_id,
            dry_run=request.dry_run,
            force=request.force
        )
        return SingleTagResponse(
            entity_id=result.get("entity_id"),
            entity_type=result.get("entity_type"),
            entity_name=result.get("entity_name"),
            status=result.get("status"),
            tags=result.get("tags")
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity tagging pipeline error: {str(e)}"
        )
