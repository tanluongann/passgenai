import json
import os
import random
import string

class GraphLearner:
    """Take a folder containing a data.txt file and build a creation model.
    The model will dump the file as model.json
    """

    STR = '#start#'
    END = "#end#"

    def __init__(self, folder='simplelist', max_passwords=500, min_password_size=8):
        """Create the learner, with empty model and data.

        Optional: 
            - min_password_size: minimum size of acceptable passwords, default is 0.
        """
        self.data = []
        self.model = {}
        self.folder = folder
        self.max_passwords = max_passwords
        self.min_password_size = min_password_size
        self.generated = []

    def set_folder(self, folder):
        """Set the data folder.

        Parameters:
            - folder: path of the folder containing the data and used to generate the model
        """
        self.folder = folder

    def load_data(self):
        """Load the learning data from the data.txt file.
        """
        full_filename = os.path.join(self.folder, 'data.txt')
        with open(full_filename, 'r') as f:
            d = f.read()
            for e in d.split('\n'):
                if len(e.strip()) > self.min_password_size:
                    self.data.append(e)

    def learn(self):
        """Perform the learning, by reading ans decomposing all words in data.txt
        """
        for w in self.data:
            l1 = GraphLearner.STR
            l2 = GraphLearner.STR
            wl = list(w)
            wl.append(GraphLearner.END)
            for i in range(len(wl)):
                l3 = wl[i]
                k = "%s%s" % (l1, l2)
                if k not in self.model:
                    self.model[k] = []
                self.model[k].append(l3)
                l1 = l2
                l2 = l3

    def save_model(self):
        """Save the model in the data folder as model.json.
        """
        full_filename = os.path.join(self.folder, 'model.json')
        with open(full_filename, 'w') as f:
            f.write(json.dumps(self.model))

    def load_model(self):
        full_filename = os.path.join(self.folder, 'model.json')
        with open(full_filename, 'r') as f:
            self.model = json.loads(f.read())

    def generate(self):
        if len(self.model) == 0:
            self.load_model()
        self.generated = []
        c = 0
        while c < 10 and len(self.generated) < self.max_passwords:
            p1 = GraphLearner.STR
            p2 = GraphLearner.STR
            e = ''
            r = []
            while e != GraphLearner.END:
                k = "%s%s" % (p1, p2)
                e = random.choice(self.model[k])
                if e != GraphLearner.END:
                    r.append(e)
                p1 = p2
                p2 = e
            final = ''.join(r)
            if final not in self.generated:
                if self.validate_password(final):
                    self.generated.append(final)
                c = 0
            else:
                c += 1
        self.save_results()

    def validate_password(self, password):
        r = len(password) >= self.min_password_size
        r = r and len(password) <= 20
        r = r and password[0] not in string.digits
        return r

    def save_results(self):
        """Save the generated words in the data folder as list.txt.
        """
        full_filename = os.path.join(self.folder, 'list.txt')
        with open(full_filename, 'w') as f:
            for e in self.generated:
                f.write(e + '\n')
