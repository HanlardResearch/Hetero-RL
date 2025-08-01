sleep 9600s
ps -ef | grep learner | grep -v grep | awk '{print $2}' | xargs kill -9