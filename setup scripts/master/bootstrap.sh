#!/usr/bin/env bash

apt-get update
apt-get install -y python3 default-jre python3-pip
# a simple two-panel file commander
apt-get install -y mc
# uncomment one of the following for graphical desktops
# NOTE: the graphical desktop is accessible through
# the main VirtualBox window (Show button)
#
# - minimal: wm & graphical server
# apt-get install -y icewm xinit xterm python3-tk
#
# - minimal desktop env: lxqt
# apt-get install -y xinit lxqt
#
# cd to the shared  directory
cd /vagrant
# python packages
pip3 install matplotlib pandas seaborn
# jupyter
pip3 install jupyter
# uncomment and modify to remove a previously installed Spark version
# rm -rf /usr/local/spark-3.0.0-preview2-bin-hadoop2.7
# remove any previously downloaded file
rm -rf spark-3.*-bin-hadoop*.tgz*
if ! [ -d /usr/local/spark-3.3.1-bin-hadoop3 ]; then
# current link as of 2022-11-28:
  wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
  tar -C /usr/local -xvzf spark-3.3.1-bin-hadoop3.tgz
  rm spark-3.3.1-bin-hadoop3.tgz
fi

if ! [ -d /usr/local/hadoop-3.3.4 ]; then
  wget https://downloads.apache.org/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz
  tar -C /usr/local -xvzf hadoop-3.3.4.tar.gz
  chown --recursive ubuntu:ubuntu /usr/local/hadoop-3.3.4
  rm hadoop-3.3.4.tar.gz
fi

if ! grep "export HADOOP_INSTALL=/usr/local/hadoop-3.3.4" /home/vagrant/.bashrc; then
  echo "export HADOOP_INSTALL=/usr/local/hadoop-3.3.4" >>  /home/vagrant/.bashrc
fi
if ! grep "export HADOOP_HOME=/usr/local/hadoop-3.3.4" /home/vagrant/.bashrc; then
  echo "export HADOOP_HOME=/usr/local/hadoop-3.3.4" >>  /home/vagrant/.bashrc
fi
if ! grep "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/" /home/vagrant/.bashrc; then
  echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/" >>  /home/vagrant/.bashrc
fi
if ! grep "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/" /usr/local/hadoop-3.3.4/etc/hadoop/hadoop-env.sh; then
  echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/" >>  /usr/local/hadoop-3.3.4/etc/hadoop/hadoop-env.sh
fi
if ! grep "export HADOOP_MAPRED_HOME=/usr/local/hadoop-3.3.4" /home/vagrant/.bashrc; then
  echo "export HADOOP_MAPRED_HOME=/usr/local/hadoop-3.3.4" >>  /home/vagrant/.bashrc
fi
if ! grep "export PYSPARK_PYTHON=/usr/bin/python3" /home/vagrant/.bashrc; then
  echo "export PYSPARK_PYTHON=/usr/bin/python3" >>  /home/vagrant/.bashrc
fi
if ! grep "export PYSPARK_DRIVER_PYTHON=jupyter" /home/vagrant/.bashrc; then
  echo "export PYSPARK_DRIVER_PYTHON=jupyter" >>  /home/vagrant/.bashrc
fi
if ! grep "export PYSPARK_DRIVER_PYTHON_OPTS=notebook" /home/vagrant/.bashrc; then
  echo "export PYSPARK_DRIVER_PYTHON_OPTS=notebook" >>  /home/vagrant/.bashrc
fi
# switch to user ubuntu
sudo -i -u ubuntu bash << EOF
echo "Switched to user ubuntu"
if ! ( echo exit | ssh localhost ) ; then
  echo "Creating keys and authorizing"
  ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
  cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
  chmod 0600 ~/.ssh/authorized_keys
fi
EOF
echo "Exited user ubuntu"
