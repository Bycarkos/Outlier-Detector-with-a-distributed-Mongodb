#!/bin/bash



mongod -shutdown --dbpath /u02/mongo/db
mongod --config /home/student/Config/Master-Slave-18.conf


for i in {1..5}
do
        scp Config/Master-Slave.conf student@mongo-${i}.grup01.bdnr:Config/Master-Slave.conf
        cat << EOF | ssh mongo-${i}.grup01.bdnr 
        mongod -shutdown --dbpath /u02/mongo/db
        mongod --config Config/Master-Slave.conf
        exit
EOF

done
	
