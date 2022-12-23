#!/bin/bash

# Creates a tarball

date=` date +%F `
version=` date +%y.%m.%d `
echo "Today: " $date

sourcefiles="src/*.py"

parfiles="par/*.par "

scripts="sh/*.sh "

others="main.py\
 README.*"

files="$sourcefiles $parfiles $scripts $others"

#output="py-ppm$version.tar.bz2"
output="py-ppm.tar.bz2"

tar cjfv $output $files

echo "File " $output " ready!"
echo

echo "-------------------------------------------"
