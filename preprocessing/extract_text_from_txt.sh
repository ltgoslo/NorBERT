# /bin/bash

# Recursively walks through all the subdirectories of the given years
# Extracts text from all txt files found, saves to a single gzipped file

find ${1} -type f -name *.txt -exec cat {} \; | python3 process_txt_nak.py | sed -e 's/<[^>]*>//g' | gzip >> ${2}.txt.gz

