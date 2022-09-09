#!/bin/bash

function Outliers(){
 cat <<EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test
use Outliers
db.New_Outliers.aggregate([
{\$facet:{TotalOutliers:[{\$count:"DocumentsTotals"}],
	  DocumentsExpermient:[{\$sortByCount: "\$label"}],
	  DocumentsConfiguracio:[{\$sortByCount: "\$conf"}]
}}
])
EOF
}

function Experiments(){
  cat << EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test
use Outliers
db.Experimental.aggregate([
    
    {\$facet:{"DocsX-K":[{\$sortByCount:"\$experiment.k"}],
            "Docs-X-TipusExperiment":[{\$sortByCount:"\$met"}],
            "DocsTotals":[{\$count:"TotalDocs"}],
            "Docs-X-Conf":[{\$sortByCount:"\$conf"}]
    }}])

EOF
}

function Iris(){
 cat << EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test

use Outliers
db.Iris.aggregate([
    {\$facet:{TotalDocuments:[{\$count:"NombreDocuments"}],
        DocumentsClasse:[{\$sortByCount:"\$label"}],
        CaracterístiquesFlor:[{\$project:{vector:1,count:{\$size:"\$vector"}}},
        {\$match:{count:{\$eq:4}}},
        {\$count:"ElementsAmbTotesCaracterístiques"}]
    }}])

EOF
}

function LetterRecognition(){
cat << EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test

use Outliers

db.LetterRecognition.aggregate([
    {\$facet:{TotalDocuments:[{\$count:"NombreDocuments"}],
        DocumentsClasse:[{\$sortByCount:"\$label"}],
        CaracterísiquesLletra:[{\$project:{vector:1,count:{\$size:"\$vector"}}},
        {\$match:{count:{\$eq:16}}},
        {\$count:"ElementsAmbTotesCaracterístiques"}]
    }}
    
    ])
EOF
}

function BreastCancer(){

cat << EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test
use Outliers
db.BreastCancer.aggregate([
    {\$facet:{TotalDocuments:[{\$count:"NombreDocuments"}],
        PersonesAmbCancer:[{\$sortByCount:"\$label"}],
        CaracterístiquesTumor:[{\$project:{vector:1,count:{\$size:"\$vector"}}},
        {\$match:{count:{\$eq:9}}},
        {\$count:"ElementsAmbTotesCaracterístiques"}],
        "GentAmbProva>1":[{\$group:{_id:"\$id", Cancer:{\$push:"\$label"}, Nproves:{\$sum:1}}},
        {\$match:{Nproves:{\$gt:1}}}]
    }}
])
EOF
}

function Ionosphere(){
cat << EOF | mongo main.grup01.bdnr:27017/admin -u TEST -p test
use Outliers

db.Ionosphere.aggregate([
    {\$facet:{TotalDocuments:[{\$count:"NombreDocuments"}],
        "#ClassesIonosphere":[{\$sortByCount:"\$label"}],
        CaracteristiquesInosphere:[{\$project:{vector:1,count:{\$size:"\$vector"}}},
        {\$match:{count:{\$eq:34}}},
        {\$count:"ElementsAmbTotesCaracterístiques"}],
    }}
    ])
EOF

}




${1}
