import fire
import os
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class TrainerMain:
    def t5(self, config: str = "t5"):
        from odd.encdec.t5 import T5Task

        T5Task.main(config)

    def t5_lm_adapted(self, config: str = "t5-lm-adapted"):
        from odd.encdec.t5 import T5LMAdaptedTask

        T5LMAdaptedTask.main(config)

    def gpt(self, config: str = "gpt"):
        from odd.gpt.gpt import GPTTask

        GPTTask.main(config)


if __name__ == "__main__":
    fire.Fire(TrainerMain)
