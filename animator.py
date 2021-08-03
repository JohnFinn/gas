from mpl_proc import MplProc, ProxyObject

class Animator:

    def __init__(self, mpl_proc: MplProc):
        self.mpl_proc = mpl_proc
        self.mpl_proc.proxy_ax.set(ylim=(0, 20), xlim=(0, 10000))

        def init(objs):
            ax = objs['ax']
            ltrain, = objs['ltrain'], = ax.plot([], [], linestyle=':', color='#EF8354')
            objs['ltest'], = ax.plot([], [], color=ltrain.get_color())
            objs['train'], objs['test'] = [], []


        self.ln_train, self.ln_test = self.mpl_proc.proxy_ax.plot([], [], [], [])
        self.proxy_train = self.mpl_proc.new_proxy([])
        self.proxy_test = self.mpl_proc.new_proxy([])
        self.cnt = 0
        self.mpl_proc.call_function(init)
        self.proxy_ltrain = ProxyObject(self.mpl_proc.conn, 'ltrain')
        self.proxy_ltest  = ProxyObject(self.mpl_proc.conn, 'ltest')

    def add(self, y1, y2):
        def update(objs, y1, y2):
            objs['train'].append(y1)
            objs['test'].append(y2)
            rng = range(len(objs['train']))
            objs['ltrain'].set_data(rng, objs['train'])
            objs['ltest'].set_data(rng, objs['test'])

        self.mpl_proc.call_function(update, y1, y2)
