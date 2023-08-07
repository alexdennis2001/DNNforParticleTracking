class SimpleProgress:
    def __init__(self, iterable, n_checkpoints=10):
        self.__iterable = list(iterable)
        self.total = len(self.__iterable)
        self.n_checkpoints = n_checkpoints
        self.n_reports = 0
        self.check = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.check >= self.n_reports*self.total/self.n_checkpoints:
            self.n_reports += 1
            print(f"{100*self.check/self.total:0.1f}% complete", flush=True)

        if self.check >= self.total:
            raise StopIteration

        item = self.__iterable[self.check]
        self.check += 1

        return item