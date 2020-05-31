#!/bin/bash

if [ $# -lt 2 ] 
then

	echo "Too few argument";
	echo "$0 <cfg_path weights_path>"

	exit 1;
fi

if [ $# -ge 2 ]
then
	
	echo "Start darknet"

#DBUS_FATAL_WARNINGS=0 ./darknet detector demo cfg/coco.data $1 $2 -c ${3-0} -fps ${4-30} -w ${5-640} -h ${6-480}
#./darknet detector demo cfg/coco.data $1 $2 -c ${3-0} -fps ${4-30} -w ${5-640} -h ${6-480}
	./darknet detector demo cfg/coco.data $1 $2 -c ${3-0} -offset ${4-0}

fi

