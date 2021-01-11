# /bin/bash

# Recursively walks through all the subdirectories of the given path
# Converts file encoding to UTF-8

find ${1} -type f -name *.xml -execdir iconv -f ISO-8859-15 -t UTF-8 -o '{}'.utf8 '{}' \;
