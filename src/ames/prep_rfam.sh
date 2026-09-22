#!/bin/bash
set -e

cd "$(dirname $0)"


rfamdb="../rfam"

rm -rf $rfamdb && mkdir $rfamdb

echo "Downloading rfam..."
wget -q https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz -O $rfamdb/Rfam.cm.gz

gunzip $rfamdb/Rfam.cm.gz

echo "Indexing rfam..."
cmpress $rfamdb/Rfam.cm

echo "Rfam.cm is ready for use"