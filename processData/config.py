# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: oriolrt
'''


class config:
    """
    Configuration class to open a connection to an Oracle server. All the parameters required for a proper connection are stored here
    """
    __slots__ = ['__username', '__password', '__sid', '__db', '__hosts', '__auth_db']

    def __init__(self, params):
        # def __init__(self, IPsHosts,port="1521",username='system',password='oracle',sid='xe'):
        if "username" in params:
            self.__username = params["username"]
        else:
            self.__username = None

        if "password" in params:
            self.__password = params["password"]
        else:
            self.__password = None

        if not "sid" in params:
            self.__sid = "ee"
        else:
            self.__sid = params["sid"]

        if not "auth_db" in params:
            self.__auth_db = None
        else:
            self.__auth_db = params["auth_db"]

        if "db" in params:
            self.__db = params["db"]
        else:
            self.__db = None

        self.__hosts = {}
        for key in params["servers"]:
            serverParams = params["servers"][key]
            self.__hosts[int(key)] = serverParams

            if "port" in serverParams:
                self.__hosts[int(key)]["port"] = int(serverParams["port"])
            else:
                self.__hosts[int(key)]["port"] = 1521

    @property
    def username(self):
        return self.__username

    @username.setter
    def username(self, u):
        self.__username = u

    @property
    def password(self):
        return self.__password

    @password.setter
    def password(self, val):
        self.__password = val

    @property
    def sid(self):
        return self.__sid

    @sid.setter
    def sid(self, val):
        self.__sid = val

    @property
    def auth_db(self):
        return self.__auth_db

    @auth_db.setter
    def auth_db(self, val):
        self.__auth_db = val

    @property
    def db(self):
        return self.__db

    @db.setter
    def db(self, val):
        self.__db = val

    @property
    def hosts(self):
        return self.__hosts

    @hosts.setter
    def hosts(self, val):
        self.__hosts = val
