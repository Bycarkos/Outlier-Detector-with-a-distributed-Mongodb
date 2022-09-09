#!/bin/bash


function InsertData(){

declare -a CollectionNames=("Iris" "Breast-Cancer" "Ionosphere" "Letter-Recognition")
declare -a files=("iris.csv" "breast-cancer-wisconsin.csv" "ionosphere.csv" "letter-recognition.csv")

for i in $(seq 0 1 3)
do 
  python3 /home/student/insertData.py -c /home/student/config.json -M /home/student/metadata.json -N "${CollectionNames[$i]}" -f /home/student/resources/UCI/"${files[$i]}" -t vector
done


 python3 /home/student/insertData.py -c /home/student/config.json -M /home/student/metadata.json -N MirFlickr25K -f /home/student/resources/MirFlickr25K/ -t image
}

 function InserOutliers(){ 
python3 testOutlierDetector.py -c configTest.json -m COD -p "{'k:1'm:1'sigma:1}" -M metadata.json 
python3 testOutlierDetector.py -c configTest.json -m DMOD -p "{'k':1}" -M metadata.json 
python3 testOutlierDetector.py -c configTest.json -m DMOD -p "{'k':3}" -M metadata.json 
python3 testOutlierDetector.py -c configTest.json -m DMOD -p "{'k':6}" -M metadata.json 
python3 testOutlierDetector.py -c configTest.json -m DMOD -p "{'k':9}" -M metadata.json 
}

if [ $# -eq 1 ]; then
       ${1}
fi       
