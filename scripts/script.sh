#!/bin/bash



read -p "Introdueix quina funció vols utilitza --> Validation or CopyFile or Activat_Flags: " value

CopyFile(){
echo "Realitzant les copies del arxiu mongodb.pem per a tots els servidors"
for i in {1..8}
do
        ssh student@mongo-${i}.grup01.bdnr "mkdir ssl Config"
        scp Config/Master-Slave.conf student@mongo-${i}.grup01.bdnr:Config/Master-Slave.conf
	scp Config/standAlone.conf student@mongo-${i}.grup01.bdnr:Config/standAlone.conf
        scp ssl/mongodb.pem student@mongo-${i}.grup01.bdnr:ssl/mongodb.pem


done

echo "Arxius Copiats"
}

Activate_Flags(){
echo "Activació dels flags"
for i in {1..8}
do
        cat << EOF | ssh mongo-${i}.grup01.bdnr 
        mongod -shutdown --dbpath /u02/mongo/db
        mongod --config Config/standAlone.conf
        exit
EOF

done
echo "Activació dels flags realitzada"
}

if [ "${value}" = "CopyFile" ]; then
CopyFile
fi


if [ "${value}" = "Activate_Flags" ]; then 
Activate_Flags
fi





