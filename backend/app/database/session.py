from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base


def _create_database_engine(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine, factory


def create_database(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    return _create_database_engine(database_url)


def create_offline_database(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    return _create_database_engine(database_url)


def initialize_database(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def initialize_offline_database(engine: Engine) -> None:
    from app.database.offline_models import OfflineBase

    OfflineBase.metadata.create_all(bind=engine)
