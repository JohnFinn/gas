from mpl_proc import MplProc, ProxyObject

class Animator:

    def __init__(self, mpl_proc: MplProc):
        self.mpl_proc = mpl_proc

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

    def add(self, y1, y2):

        # want this to be like that
        # with self.mpl_proc.buffer_calls():
            # self.proxy_train.append(y1)
            # self.proxy_test.append(y2)
            # rng = range(range(len(self.proxy_train)))
            # self.proxy_ltrain.set_data(rng, self.proxy_train)
            # self.proxy_ltest.set_data(rng, self.proxy_test)

        def update(objs, y1, y2):

            '''
            wanted this to work TODO findout why doesn't work
            def foo(train, test, ltrain, ltest, y1, y2, **_):

                train.append(y1)
                test.append(y2)
                rng = range(len(train))
                ltrain.set_data(rng, train)
                ltest.set_data(rng, test)

            return foo(**objs, y1=y1, y2=y2)
            '''

            train = objs['train']
            test = objs['test']
            ltrain = objs['ltrain']
            ltest = objs['ltest']

            train.append(y1)
            test.append(y2)
            rng = range(len(train))
            ltrain.set_data(rng, train)
            ltest.set_data(rng, test)

            ax = objs['ax']
            ax.relim()
            ax.autoscale_view(True, True, True)

        self.mpl_proc.call_function(update, y1, y2)
