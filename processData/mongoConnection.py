# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: Oriol Ramos Terrades
'''

from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient, errors
from getpass import getpass

try:
    from .noConnection import noConnection as nc
except ImportError:
    from . import noConnection as nc


class mongoConnection(nc):
    """

    """

    def __init__(self, cfg):
        '''
            Constructor
        '''

        super(mongoConnection, self).__init__(cfg)

    @property
    def bd(self):
        if not self.isStarted:
            self.startSession()

        return self.__noConnection_bd

    @bd.setter
    def bd(self, db):
        self.__noConnection_bd = db

    @property
    def connectDB(self):
        """
          Connect to a oracle server given the connexion information saved on the cfg member variable.

          :return: None
        """
        cfg = self.cfg
        # numServers = len(cfg.hosts)
        user_str = ""

        if 'auth_db' in cfg:
            auth_db = '/' + cfg.auth_db
        else:
            auth_db = ''

        if 'username' in cfg.keys():
            if 'password' not in cfg.keys():
                cfg['password'] = getpass("Password de l'usuari {} de MongoDB: ".format(cfg.username))
            user_str = "{}:{}@".format(cfg['username'], cfg['password'])

        if "ssh" in cfg:
            sshParams = cfg["ssh"]

            DSN = "mongodb://{}localhost:{}{}".format(user_str, sshParams["port"], auth_db)

            if "password" in sshParams:
                if sshParams["password"] == "":
                    sshParams["password"] = getpass(
                        "Password de l'usuari {} a {}: ".format(sshParams["username"], cfg["hostname"]))

            self.server = SSHTunnelForwarder((sshParams["hostname"], int(sshParams["port"])),
                                             ssh_username=sshParams["username"],
                                             ssh_password=sshParams["password"],
                                             remote_bind_address=(cfg["hostname"], cfg["port"]),
                                             local_bind_address=("", int(sshParams["port"]))
                                             )
            self.server.start()
        else:
            DSN = "mongodb://{}{}:{}{}".format(user_str, cfg["hostname"], cfg["port"], auth_db)

        try:
            self.conn = MongoClient(DSN, serverSelectionTimeoutMS=100)
            self.conn.server_info()  # force connection on a request as the  # connect=True parameter of MongoClient seems  # to be useless here
            self.isStarted = True
        except errors.ServerSelectionTimeoutError as err:
            self.isStarted = False
            # do whatever you need
            print(err)

        if 'bd' in cfg:
            self.bd = self.conn[cfg['bd']]

        return self.conn

    def close(self):
        self.conn.close()
        if self.server is not None: self.server.stop()

    def startSession(self):
        self.connectDB
        return self.isStarted

    def testConnection(self):

        dbs = self.conn.database_names()

        print("Databases: {}".format(" ".join(dbs)))

        return True
