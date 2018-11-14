class NAC:
    def __init__(self, o_space, a_space, model):
        self.o_space = o_space
        self.a_space = a_space
        self.model = model(o_space, a_space)

    def __call__(self, o_space, log_prob=False):
        if log_prob:
            return self.model(o_space)
        else:
            return self.model(o_space)[0]
