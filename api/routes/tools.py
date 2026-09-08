from fastapi import APIRouter, HTTPException
from registry.store import get_tool, list_tools

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def get_tools():
    return list_tools(status="success")


@router.get("/{tool_id}")
async def get_tool_by_id(tool_id: str):
    tool = get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool
