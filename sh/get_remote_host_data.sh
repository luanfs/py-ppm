#!/bin/bash
# Luan Santos - 2023
# Script to get data from remote ime usp host servers.

#-------------------------------------------------------------------------------------------------------
# Hosts and users
remote_host1="brucutuiv.ime.usp.br"  # accessed from local host
remote_host2="ybytu.ime.usp.br"      # accessed from remote host1
user1="luansantos"                   # user at remote_host1
user2="luansantos"                   # user at remote_host2
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Directories and output
output_dir_remote_host1="/var/tmp/"                              #remote host data directory (remote_host1)
output_dir_remote_host2="/home/luansantos/doc/ppm/"              #remote host data directory (remote_host2)
output_dir_local_host="/home/luan/doc/code/py-ppm/"  #local data directory
data='graphs'           #name of directory where data is in output_dir_remote_host2
output='ppm_data'   #output file (.tar)

#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Fisrt, we create a tarball with the target data in remote host 2
#-------------------------------------------------------------------------------------------------------
echo "Creating tarball at $remote_host2:"
ssh -t $user1@$remote_host1 "ssh -t $user2@$remote_host2 <<EOF
	cd $output_dir_remote_host2;
	tar cjfv $output $data
EOF"
echo "Created tarball at $remote_host2"
echo

#-------------------------------------------------------------------------------------------------------
# Get data from remote_host2 to remote_host1
echo "Getting data from $remote_host2 to $remote_host1:"
ssh -t $user1@$remote_host1 "scp -r $user2@$remote_host2:$output_dir_remote_host2$output $output_dir_remote_host1/"
echo "Got data from $remote_host2 to $remote_host1"
echo
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Get data from remote_host1 to local
echo "Getting data from $remote_host1 to local:"
echo $output_dir_remote_host1$output 
echo $output_dir_local_host
scp -r $user1@$remote_host1:$output_dir_remote_host1$output $output_dir_local_host
echo "Got data from $remote_host1 to local"
echo
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Clean data from remote_host1
echo "Remove data from $remote_host1:"
ssh -t $user1@$remote_host1 "cd $output_dir_remote_host1; rm -rf  $output" 
echo "Removed data from $remote_host1"
echo
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# Clean data from remote_host2
echo "Remove data from $remote_host2:"
ssh -t $user1@$remote_host1 "ssh -t $user2@$remote_host2 rm -rf $output_dir_remote_host2$output"
echo "Removed data from $remote_host2"
echo
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------
# untar the file and clean
date=` date +%F `
version=` date +%y.%m.%d `
echo "Untar and clean"
cd ..
tar -xvf $output
mv $data $remote_host2$data$version
rm -rf $output
#-------------------------------------------------------------------------------------------------------
