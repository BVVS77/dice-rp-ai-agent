import argparse

from sqlalchemy.exc import SQLAlchemyError

from dicerp.config import Settings
from dicerp.db import seed_database
from dicerp.errors import ProviderUnavailable
from dicerp.providers import open_provider
from dicerp.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DiceRP's local demo data.")
    parser.add_argument("command", choices=["seed", "index"])
    args = parser.parse_args()
    try:
        settings = Settings()
        if args.command == "seed":
            seed_database(settings.dicerp_state_dir / "snapshot.sqlite3")
            print("Created the synthetic SQLite snapshot.")
        else:
            with open_provider(settings) as provider:
                build_index(
                    settings.dicerp_state_dir / "rules.json",
                    settings.embedding_model,
                    provider.embedder,
                    provider=settings.dicerp_ai_provider,
                )
            print("Indexed the sample handbook.")
    except FileExistsError:
        parser.exit(1, "Snapshot already exists. Set DICERP_STATE_DIR to seed another copy.\n")
    except ProviderUnavailable as exc:
        parser.exit(1, f"Indexing failed. {exc}\n")
    except (OSError, SQLAlchemyError, ValueError):
        parser.exit(
            1,
            "Setup failed. Check the state directory, provider and model settings, "
            "and API key (OpenAI only).\n",
        )


if __name__ == "__main__":
    main()
