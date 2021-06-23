import matplotlib

class FigRecord:

    def __init__(self, fig: matplotlib.figure.Figure, name: str, filename: str):
        self.fig = fig
        self.name = name
        self.filename = filename

    def to_markdown(self):
        self.fig.savefig(self.filename)
        return f'![{self.name}]({self.filename})'


class StringRecord:

    def __init__(self, string: str):
        self.string = string

    def to_markdown(self):
        return self.string


class Reporter:

    def __init__(self, filename: str):
        self.filename = filename
        self.records = []

    def append(self, record):
        self.records.append(record)

    def to_markdown(self):
        return '\n'.join([r.to_markdown() for r in self.records]) + '\n\n'

    def write(self):
        with open(self.filename, 'at') as f:
            f.write(self.to_markdown())
