import numpy as np

from pyics import Model
from collections import Counter


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""
    allChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n < k:
        return str(allChars[n])
    else:
        return str(decimal_to_base_k(n // k, k) + allChars[n % k])

def save_plot(x, y):
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    graph1 = plt.plot(x, y)
    plt.title("repit distribution")
    plt.ylabel('Num of repits')
    plt.xlabel('Rule')

    plt.savefig('repit_distribution', fmt='png', dpi=100)

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 10)
        self.make_param('height', 10)
        self.make_param('rule', 34, setter=self.setter_rule)
        self.build_rule_set()

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        self.rule_set_size = self.k ** (2 * self.r + 1)
        self.max_rule_number = self.k ** self.rule_set_size
        return max(0, min(val, self.max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""

        numberInCurrentBase = list(decimal_to_base_k(self.rule, self.k))

        self.rule_set = [0]*(self.rule_set_size - len(numberInCurrentBase))
        self.rule_set += numberInCurrentBase

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        inp = "".join([str(int(x)) for x in inp])
        num = int(inp, self.k)
        pos = self.rule_set_size - num - 1
        return self.rule_set[pos]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        value_array = [i for i in range(0, self.k)]
        return np.random.choice(value_array, self.width)

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)

    def calculateRepits(self, result):
        hashArray = []
        for x in result:
            inp = "".join([str(int(d)) for d in x])
            num = int(inp, self.k)
            hashArray.append(num)

        counter = Counter(hashArray)
        averge = 0

        for x in counter:
            averge += counter[x]

        averge = averge/len(counter)

        #print counter, averge
        return averge

if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI

    averges_array = []
    x_scale_array = [i for i in range(0, 255)]

    for i in range(0, 255):
        mod = CASim()
        mod.make_param('rule', i)
        mod.build_rule_set()
        mod.t = 0
        mod.reset()
        while mod.t < mod.height:
            mod.step()
        x = mod.calculateRepits(mod.config)
        averges_array.append(x)

    save_plot(x_scale_array, averges_array)

    cx = GUI(sim)
    cx.start()





