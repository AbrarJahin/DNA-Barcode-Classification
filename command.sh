# Run a long running Command
cd /home/ajahin/contest/ && git pull && cd ~
nohup python3 /home/ajahin/contest/main.py>/home/ajahin/output.txt &
tail -f /home/ajahin/output.txt

#See List
ps xw

#Kill a process

kill -9 <PId>
