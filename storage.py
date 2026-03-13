import json
import os

class Storage:
    def __init__(self, filename='contacts.json'):
        self.filename = filename
        self.data = {}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.data = json.load(f)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)