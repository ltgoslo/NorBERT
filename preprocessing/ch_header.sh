# /bin/bash

# Recursively walks through all the subdirectories of the given path
# Converts XML headers to UTF-8

find ${1} -type f -name *.xml -execdir sed -i -e s'/encoding=\"ISO-8859-1\"/encoding=\"utf-8\"/' {} \;
