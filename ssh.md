# To tunnel between host and remote computer
~~~
ssh -L 5902:localhost:5902 -C -l username remote_computer
~~~
# To get a list of running servers
~~~
vncserver -list
~~~

# SCP
## remote to local 
~~~
scp <remote_username>@<IPorHost>:<PathToFile>   <LocalFileLocation>
~~~
## local to remote
~~~
scp test.txt userbravo@destination:/location2
~~~
