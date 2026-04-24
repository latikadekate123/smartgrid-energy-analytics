import os
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass
class Settings:
    pg_host: str = os.getenv("PGHOST", "postgres")
    pg_port: int = int(os.getenv("PGPORT", "5432"))
    pg_db: str = os.getenv("PGDATABASE", "smartgrid")
    pg_user: str = os.getenv("PGUSER", "smartgrid")
    pg_password: str = os.getenv("PGPASSWORD", "smartgrid")
    raw_data_path: str = os.getenv("RAW_DATA_PATH", "")
    model_output_path: str = os.getenv("MODEL_OUTPUT_PATH", "models/gradient_boosted.joblib")

    @property
    def sqlalchemy_url(self) -> str:
        user = quote_plus(self.pg_user)
        password = quote_plus(self.pg_password)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )
