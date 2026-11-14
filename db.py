# vowelpipe/db.py
from datetime import datetime
import pandas as pd
from sqlalchemy import (create_engine, select, ForeignKey, String, Boolean,
                        JSON, UniqueConstraint, Index)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from . import config as cfg

class Base(DeclarativeBase):
    pass

class Speaker(Base):
    __tablename__ = 'speakers'
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)
    sex: Mapped[str] = mapped_column(String(16))
    dialect: Mapped[str | None] = mapped_column(String(128))
    age_group: Mapped[str | None] = mapped_column(String(32))
    consent: Mapped[bool] = mapped_column(Boolean, default=False)

class Recording(Base):
    __tablename__ = 'recordings'
    id: Mapped[int] = mapped_column(primary_key=True)
    speaker_id: Mapped[int] = mapped_column(ForeignKey('speakers.id'))
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    duration_s: Mapped[float]
    sample_rate: Mapped[float]
    n_channels: Mapped[int]
    textgrid_file: Mapped[str | None] = mapped_column(String(255))

class AnalysisRun(Base):
    __tablename__ = 'analysis_runs'
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    params: Mapped[dict] = mapped_column(JSON)
    versions: Mapped[dict] = mapped_column(JSON)

class Token(Base):
    __tablename__ = 'tokens'
    __table_args__ = (Index('ix_tokens_run_vowel', 'run_id', 'vowel'),
                      Index('ix_tokens_speaker', 'speaker_id'))
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('analysis_runs.id'))
    recording_id: Mapped[int] = mapped_column(ForeignKey('recordings.id'))
    speaker_id: Mapped[int] = mapped_column(ForeignKey('speakers.id'))
    vowel: Mapped[str] = mapped_column(String(16))
    word: Mapped[str | None] = mapped_column(String(64))
    prev_phone: Mapped[str | None] = mapped_column(String(16))
    next_phone: Mapped[str | None] = mapped_column(String(16))
    start_s: Mapped[float]
    end_s: Mapped[float]
    duration_s: Mapped[float]
    f0: Mapped[float | None]
    f1: Mapped[float | None]
    f2: Mapped[float | None]
    f3: Mapped[float | None]
    ceiling_hz: Mapped[float]
    qc: Mapped[str] = mapped_column(String(16), default='')
    keep: Mapped[bool] = mapped_column(Boolean, default=True)

class TokenPoint(Base):
    __tablename__ = 'token_points'
    id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[int] = mapped_column(ForeignKey('tokens.id'), index=True)
    proportion: Mapped[float]
    f1: Mapped[float | None]
    f2: Mapped[float | None]
    f3: Mapped[float | None]

class NormalisedValue(Base):
    __tablename__ = 'normalised_values'
    __table_args__ = (UniqueConstraint('token_id', 'method'),)
    id: Mapped[int] = mapped_column(primary_key=True)
    token_id: Mapped[int] = mapped_column(ForeignKey('tokens.id'), index=True)
    method: Mapped[str] = mapped_column(String(16))
    f1n: Mapped[float | None]
    f2n: Mapped[float | None]

def default_engine():
    cfg.OUT_DIR.mkdir(exist_ok=True)
    return create_engine('sqlite:///' + str(cfg.OUT_DIR / 'vowels.db'))
    # PostgreSQL: create_engine('postgresql+psycopg2://user:password@host/dbname')

def _f(x):
    return None if pd.isna(x) else float(x)

def ingest(engine, speakers, recordings, tokens, norm_methods, params, versions):
    '''Write one analysis run to the database.
    speakers   : DataFrame(speaker, sex, dialect, age_group, consent)
    recordings : DataFrame(speaker, filename, duration, sample_rate, n_channels, textgrid)
    tokens     : pipeline DataFrame (needs speaker, recording, qc, keep, f*_20/50/80)
    norm_methods : {'lobanov': ('f1_lob', 'f2_lob'), ...} columns present in tokens'''
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        run = AnalysisRun(params=params, versions=versions)
        s.add(run)
        spk = {}
        for r in speakers.itertuples(index=False):
            sp = s.scalar(select(Speaker).where(Speaker.code == r.speaker))
            if sp is None:
                sp = Speaker(code=r.speaker, sex=r.sex, dialect=r.dialect,
                             age_group=r.age_group,
                             consent=str(r.consent).strip().lower() == 'yes')
                s.add(sp)
            spk[r.speaker] = sp
        s.flush()
        rec = {}
        for r in recordings.itertuples(index=False):
            rc = s.scalar(select(Recording).where(Recording.filename == r.filename))
            if rc is None:
                rc = Recording(speaker_id=spk[r.speaker].id, filename=r.filename,
                               duration_s=r.duration, sample_rate=r.sample_rate,
                               n_channels=r.n_channels, textgrid_file=r.textgrid)
                s.add(rc)
            rec[r.filename] = rc
        s.flush()

        for t in tokens.itertuples(index=False):
            tok = Token(run_id=run.id, recording_id=rec[t.recording].id,
                        speaker_id=spk[t.speaker].id, vowel=t.vowel, word=t.word,
                        prev_phone=t.prev_phone, next_phone=t.next_phone,
                        start_s=t.start, end_s=t.end, duration_s=t.duration,
                        f0=_f(t.f0), f1=_f(t.f1), f2=_f(t.f2), f3=_f(t.f3),
                        ceiling_hz=t.ceiling_hz, qc=t.qc, keep=bool(t.keep))
            s.add(tok)
            s.flush()
            for tag in (20, 50, 80):
                s.add(TokenPoint(token_id=tok.id, proportion=tag / 100.0,
                                 f1=_f(getattr(t, f'f1_{tag}')),
                                 f2=_f(getattr(t, f'f2_{tag}')),
                                 f3=_f(getattr(t, f'f3_{tag}'))))
            for method, (c1, c2) in norm_methods.items():
                a, b = _f(getattr(t, c1)), _f(getattr(t, c2))
                if a is not None and b is not None:
                    s.add(NormalisedValue(token_id=tok.id, method=method, f1n=a, f2n=b))
        s.commit()
        return run.id

def tokens_dataframe(engine, run_id, keep_only=True, method='lobanov'):
    '''Tokens of one run with demographic columns and normalised F1/F2.'''
    stmt = (select(Token.id, Speaker.code.label('speaker'), Speaker.sex,
                   Speaker.dialect, Token.vowel, Token.word, Token.f0,
                   Token.f1, Token.f2, Token.f3, Token.qc, Token.keep,
                   NormalisedValue.f1n, NormalisedValue.f2n)
            .select_from(Token)
            .join(Speaker, Token.speaker_id == Speaker.id)
            .join(NormalisedValue,
                  (NormalisedValue.token_id == Token.id) & (NormalisedValue.method == method),
                  isouter=True)
            .where(Token.run_id == run_id))
    if keep_only:
        stmt = stmt.where(Token.keep.is_(True))
    return pd.read_sql(stmt, engine)
