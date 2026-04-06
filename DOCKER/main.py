from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()
DATA_FILE = Path("data.txt")
DATA_FILE.touch(exist_ok=True)

class Item(BaseModel):
    value: str


def read_items() -> list[str]:
    return [line.strip() for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_items(items: list[str]) -> None:
    DATA_FILE.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


@app.get("/")
async def root():
    return {
        "message": "App is running.",
        "endpoints": ["GET /items", "POST /items", "DELETE /items"],
        "docs": "/docs"
    }


@app.get("/items")
async def list_items():
    return {"items": read_items()}


@app.post("/items", status_code=status.HTTP_201_CREATED)
async def add_item(item: Item):
    items = read_items()
    if item.value in items:
        raise HTTPException(status_code=400, detail="Item already exists")
    items.append(item.value)
    write_items(items)
    return {"added": item.value}


@app.delete("/items")
async def remove_item(item: Item):
    items = read_items()
    if item.value not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    items = [value for value in items if value != item.value]
    write_items(items)
    return {"removed": item.value}
