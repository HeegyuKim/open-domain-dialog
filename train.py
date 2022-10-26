import fire
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class TrainerMain:
    def t5(self, config: str = "t5"):
        from odd.encdec.t5 import T5Task

        T5Task.main(config)


if __name__ == "__main__":
    fire.Fire(TrainerMain)