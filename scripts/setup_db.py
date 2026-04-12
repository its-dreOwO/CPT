import sys
import os

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.database import Base, engine


def setup():
    print("Creating all tables in the database...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    setup()
