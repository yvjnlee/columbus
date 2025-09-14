import dspy

from columbus.config.settings import Config


class Planner:
    def __init__(self, model: str, cfg: Config):
        self.model = model
        self.cfg = cfg

    def __start__(self):
        lm = dspy.LM(self.model, api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=lm)

    def plan(self, task: str):
        return self.model.plan(task)
