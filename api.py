# vowelpipe/api.py      run with:  uvicorn vowelpipe.api:app --reload
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from . import config as cfg
from .db import default_engine, tokens_dataframe
from .plots import interactive_vowel_space

app = FastAPI(title='Vowel pipeline API')
engine = default_engine()

@app.get('/analyses/{run_id}/tokens')
def get_tokens(run_id: int, vowel: str | None = None, speaker: str | None = None,
               keep_only: bool = True, method: str = 'lobanov'):
    df = tokens_dataframe(engine, run_id, keep_only, method)
    if vowel:
        df = df[df['vowel'] == vowel]
    if speaker:
        df = df[df['speaker'] == speaker]
    if df.empty:
        raise HTTPException(status_code=404, detail='No tokens found')
    return df.astype(object).where(df.notna(), None).to_dict(orient='records')

@app.get('/analyses/{run_id}/chart', response_class=HTMLResponse)
def get_chart(run_id: int, method: str = 'lobanov'):
    df = tokens_dataframe(engine, run_id, True, method).dropna(subset=['f1n', 'f2n'])
    if df.empty:
        raise HTTPException(status_code=404, detail='No normalised tokens for this method')
    path = cfg.OUT_DIR / f'chart_{run_id}_{method}.html'
    interactive_vowel_space(df, 'f2n', 'f1n', path)
    return HTMLResponse(path.read_text(encoding='utf-8'))
