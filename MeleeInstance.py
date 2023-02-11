class MeleeInstance():
    def __init__(self, gid, model_name, doesRender, char, port, stage):
        self.gid = gid
        self.hasOpponent = False
        self.opponentId = None
        self.model_name = model_name
        self.pid = None
        self.doesRender = doesRender
        self.char = char
        self.port = port
        self.isDone = False
        self.waitingToReset = True,
        self.stage = stage
        self.prev_obs1 = None
        self.prev_obs2 = None