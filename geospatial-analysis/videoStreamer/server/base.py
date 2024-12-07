from flask import request
from flask_restful import Resource


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Base(Resource, metaclass=Singleton):
    def __init__(self):
        self.request = request
        print(self.request.url_rule.endpoint)
        self.action = self.request.url_rule.endpoint

    def get(self, action):
        self.request = request
        self.action = action

    def post(self, action):
        self.request = request
        self.action = action
        return self.invoke()

    def invoke(self):
        pass
