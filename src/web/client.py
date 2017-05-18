import socket
import os
import sys

s = socket.socket()
host = socket.gethostname()
port = 9000
s.connect((host, port))
filename = sys.argv[1]
size = len(filename)
size = bin(size)[2:].zfill(16)
s.send(size)
s.send(filename)

filesize = os.path.getsize(filename)
filesize = bin(filesize)[2:].zfill(32)
s.send(filesize)

file_to_send = open(filename, 'rb')

l = file_to_send.read()
s.sendall(l)
file_to_send.close()
print 'File Sent'

result = s.recv(32)
s.close()

print result