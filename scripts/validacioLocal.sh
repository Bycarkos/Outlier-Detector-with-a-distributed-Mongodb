#!/bin/bash

name=$1;
opcio=$3;
password="sys";

function Privileges(){

if [ "${name}" = "GestorDades" ]; then
 echo "Aquestos són els privilegis de Gestor de Dades"
 cat << EOF | mongo dcccluster.uab.es:52261/admin -u sys -p sys
use admin
db.getUser("GestorDades")
db.getRole("CRUD", {showPrivileges:true, showBuiltinRoles: true})
EOF

fi

if [ "${name}" = "TEST" ]; then

cat << EOF | mongo dcccluster.uab.es:52261/admin -u sys -p sys
use admin
db.getUser("TEST")
db.getRole("BASIC", {showPrivileges:true, showBuiltinRoles: true})
EOF
fi

}

function Find(){
cat << EOF | mongo dcccluster.uab.es:52261/admin -u ${name} -p ${password} 
use Outliers 
db.Iris.aggregate([{\$group:{"_id":"$label", "Elements_class":{\$sum:1}}}])
EOF
}

function Insert(){
  cat << EOF | mongo dcccluster.uab.es:52261/admin -u ${name} -p ${password} 
use Outliers 
db.test.insertMany([{name: "Carlos", lastname: "Boned", niu: "1533190"},{name:"Miguel", lastname: "Esteban", niu:"1530926"}])

EOF
}
function Remove(){
  cat  << EOF | mongo dcccluster.uab.es:52261/admin -u ${name} -p ${password} 
use Outliers
db.test.remove({name: "Carlos"})

EOF
}

function Update(){
  cat  << EOF | mongo dcccluster.uab.es:52261/admin -u ${name} -p ${password} 
use Outliers
db.test.update({name:"Carlos"},{\$set:{secondLastName:"Riera"}})

EOF
}

if [ $# -eq 0 ]; then
echo "No has introduit ningun parametre"
fi
if [ $# -eq 1 ]; then
Privileges

elif [ $# -eq 2 ]; then
password=${2};
echo "Les bases de Dades a les que te acces aquest usuari son: "
  cat  << EOF | mongo dcccluster.uab.es:52261/admin -u ${name} -p ${password} 
show dbs
use Outliers
db.getCollectionNames()
EOF
elif [ $# -eq 3 ];then 
password=${2};
"${opcio}"
fi




