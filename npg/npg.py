class NPG:
    def __init__(self, o_space, a_space, model):
        self.o_space = o_space
        self.a_space = a_space
        self.model = model(o_space, a_space)

    def __call__(self, o_space):
        return self.model(o_space)

