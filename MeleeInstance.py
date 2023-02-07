class MeleeInstance():
    def __init__(self, gid, model_name, doesRender, char, port):
        self.gid = gid
        self.hasOpponent = False
        self.opponentId = None
        self.model_name = model_name
        self.pid = None
        self.doesRender = doesRender
        self.char = char
        self.port = port
        self.isDone = False
        self.waitingToReset = True