# /bin/bash

# Recursively walks through all the subdirectories of the given years
# Extracts text from all XML files found, saves to a single gzipped file per each year/language.

for i in {2012..2018}

    do
	echo ${i}
	find NAK/${i}/nno -type f -name *.xml -exec python3 preprocessing/process_xml_nak.py {} \; | gzip >> ${i}_nno.txt.gz
	find NAK/${i}/nob -type f -name *.xml -exec python3 preprocessing/process_xml_nak.py {} \; | gzip >> ${i}_nob.txt.gz
    done

