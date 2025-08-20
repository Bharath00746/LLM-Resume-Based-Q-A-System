from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.rag import build_or_load_index, grounded_answer

app = FastAPI(title="Resume QA Bot", version="1.0.0")

# Build/load index at startup
index = build_or_load_index(force_rebuild=False)

# Templates & static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- Home Page ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- Ask Endpoint ----------------

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    result = grounded_answer(index, question)
    answer = result["answer"]
    # Fallback for generic questions
    if answer.strip().lower() == "i cannot find that in my resume.":
        answer = (
            "I'm an AI Resume QA Bot. Please ask questions related to the resume content."
        )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": answer
        }
    )