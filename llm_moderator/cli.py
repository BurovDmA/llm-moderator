import fire

from llm_moderator.data.download import download_data
from llm_moderator.training.train import main as train_main
from llm_moderator.utils.logging import configure_logging


class CLI:
    def download(self) -> None:
        download_data()

    def train(self) -> None:
        train_main()


def run() -> None:
    configure_logging()
    fire.Fire(CLI)


if __name__ == "__main__":
    run()
