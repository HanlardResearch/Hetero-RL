sleep 9600s
ps -ef | grep sampler | grep -v grep | awk '{print $2}' | xargs kill -9