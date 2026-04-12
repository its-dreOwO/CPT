import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import Settings

settings = Settings()

db_url = settings.DATABASE_URL

# If SQLite, ensure we don't crash because the 'data/' folder is missing
if db_url.startswith("sqlite"):
    db_path = db_url.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    # Adding check_same_thread=False for SQLite in multithreaded (FastAPI) environments
    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
else:
    engine = create_engine(db_url, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_session():
    """
    Provide a transactional scope around a series of operations.
    Usage:
        with get_session() as session:
            session.add(obj)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
