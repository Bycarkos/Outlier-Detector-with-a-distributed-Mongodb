#!/bin/bash

create_dba()
{

user=$1
pass=$2

echo "You're in machine: $(hostname)"
cat << EOF | mongo
use admin;
db.dropUser("${user}")
db.createUser(
{
	user: "${user}",
	pwd: "${pass}",
	roles: [{"role":"root","db":"admin"}],
}
)
EOF

}

for s in $(seq 3 8);
do
ssh mongo-${s}.grup01.bdnr "$(typeset -f $1); $@" < /dev/null
done
