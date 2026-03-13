import json
import os
from storage import Storage

class ContactsManager:
    def __init__(self):
        self.storage = Storage()

    def add_contact(self, name, phone, email):
        self.storage.data[name] = {'phone': phone, 'email': email}
        self.storage.save()

    def list_contacts(self):
        return list(self.storage.data.values())

    def search_contact(self, query):
        results = []
        for contact in self.storage.data.values():
            if query.lower() in contact['phone'].lower() or query.lower() in contact['email'].lower() or query.lower() in contact['name'].lower():
                results.append(contact)
        return results

    def delete_contact(self, name):
        if name in self.storage.data:
            del self.storage.data[name]
            self.storage.save()